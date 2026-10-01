//! Auto-solving of practical (trainer) tasks: a page template completely
//! separate from the theory-lesson `theory-viewer__block_type_*` system
//! handled in `main.rs`. These pages show a task description, a Monaco code
//! editor, and a check button; the check shows a `.notification_type_error`
//! or `.notification_type_success` notification once it's done.
//!
//! The code editor is Monaco (`window.monaco` is a global on these pages),
//! but file contents are written by clicking into the editor and sending
//! real keystrokes (`Element::send_keys`) rather than calling
//! `model.setValue()` through Monaco's own JS API. That's not a style
//! choice: this trainer type reloads its preview/check from a
//! server-persisted copy of the file (confirmed live — the preview iframe
//! fetches `script.js` over the network), and that copy is only updated by
//! whatever save hook the app wires to real input events.
//! `model.setValue()` still updates what's visibly in the editor (so it
//! *looks* like it worked) but never reaches that save hook, so the check
//! always ran against stale content — confirmed live by submitting the
//! exact correct answer via `setValue()` and getting the same generic
//! failure every time, then getting `notification_type_success` for the
//! identical content typed as real keystrokes instead.
//!
//! Typing has to target two different elements to work: clicking
//! `.view-lines` (what's visually rendered) is what focuses Monaco's
//! hidden `textarea.inputarea` — but `Element::send_keys` refuses to send
//! keys *to* `.view-lines` itself ("not reachable by keyboard", it's a
//! plain non-interactive `<div>`), so the actual `send_keys` calls target
//! `.inputarea` once that click has focused it.

use std::error::Error;
use std::time::Duration;

use fantoccini::elements::Element;
use fantoccini::key::Key;
use fantoccini::{Client, Locator};
use tokio::time::sleep;

use crate::ollama;

const CHECK_TASK_BUTTON_SELECTOR: &str = "[data-test-id='check-task-button']";
const TASK_DESCRIPTION_SELECTOR: &str = ".task-description";

/// After an in-app (SPA) navigation to the next task, the check button and
/// hint button mount immediately, but the task's own text is fetched
/// asynchronously and `.task-description` sits empty for a bit —
/// confirmed live sitting empty for well over `POST_CLICK_SETTLE_DELAY`
/// (1s) after a real "Продолжить"-style click. Reading task text, the
/// hint, or files on a fixed timer instead of this signal risks acting on
/// a still-loading (or, worse, still the *previous* task's) page —
/// confirmed live getting the previous task's hint text verbatim this way,
/// which fed the solver a wrong requirement it then couldn't pass no
/// matter how many attempts. `wait_for_task_description` blocks the whole
/// solve attempt on this becoming non-empty first.
const TASK_READY_TIMEOUT: Duration = Duration::from_secs(8);
const TASK_READY_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Some lessons show their theory content as a popup layered *on top of*
/// the practice-task page (confirmed live — same URL has both
/// `[data-test-id="check-task-button"]` and `theory-viewer__block_type_*`
/// content at once) rather than as its own separate page. The editor and
/// check button are there in the DOM but not actually usable while this
/// is open — attempting to solve then fails every time ("no editor
/// found", "couldn't click the check button"). `theory_popup_open` lets
/// the caller hold off until it's closed.
const THEORY_POPUP_SELECTOR: &str = ".theory-panel__theory_visible";

/// Matches both `.notification_type_error` and `.notification_type_success`
/// — same markup either way, just a different class modifier, checked in
/// `wait_for_result`. Deliberately excludes `.notification_type_hint`
/// (see `HINT_NOTIFICATION_SELECTOR`) — `get_hint` closes that one itself
/// right after reading it, so this dismiss/wait logic never has to
/// distinguish "the hint is still open" from "a check result appeared".
const NOTIFICATION_SELECTOR: &str = "[class*='notification_type_error'], [class*='notification_type_success']";
const NOTIFICATION_ERROR_CLASS: &str = "notification_type_error";
const NOTIFICATION_SUCCESS_CLASS: &str = "notification_type_success";
const NOTIFICATION_CONTENT_SELECTOR: &str = ".notification__content";
const NOTIFICATION_CLOSE_SELECTOR: &str = ".notification__close";

/// How long to wait, after clicking the check button, for a result
/// notification to appear.
const RESULT_POLL_TIMEOUT: Duration = Duration::from_secs(10);
const RESULT_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// The task hint uses the same `.notification`/`.notification__content`
/// markup as the check result (just `_type_hint` instead of
/// `_type_error`/`_type_success`) — confirmed live via a page-example with
/// the hint open. Its button toggles between "Подсказка" (closed) and
/// "Скрыть" (open, `hint-button_active` class present).
const HINT_BUTTON_SELECTOR: &str = "[data-test-id='trainer-hint-button']";
const HINT_BUTTON_ACTIVE_CLASS: &str = "hint-button_active";
const HINT_NOTIFICATION_SELECTOR: &str = ".notification_type_hint";
const HINT_APPEAR_TIMEOUT: Duration = Duration::from_secs(5);

pub struct PracticeConfig {
    pub model: String,
    pub max_attempts: u32,
}

/// A page is a practice task if it has a check button — this template has
/// nothing in common with the theory-viewer block classes, so this is a
/// separate, unrelated detection path from `try_click_next_button`.
pub async fn is_practice_task(client: &Client) -> bool {
    client
        .find(Locator::Css(CHECK_TASK_BUTTON_SELECTOR))
        .await
        .is_ok()
}

/// See `THEORY_POPUP_SELECTOR` — true while the theory content popup is
/// covering the editor/check button.
pub async fn theory_popup_open(client: &Client) -> bool {
    client
        .find(Locator::Css(THEORY_POPUP_SELECTOR))
        .await
        .is_ok()
}

/// Outcome of a single click-check-and-wait cycle, folding in the two ways
/// the click itself can fail alongside `CheckOutcome` — lets every call
/// site (the free pre-check below and the real attempt loop) handle all
/// four outcomes uniformly instead of duplicating the click/dismiss
/// boilerplate.
enum CheckAttempt {
    Result(CheckOutcome),
    ButtonMissing,
    ClickFailed,
}

/// Dismisses any leftover notification, clicks the check button, and waits
/// for a result.
async fn click_check_and_wait(client: &Client) -> CheckAttempt {
    dismiss_notification(client).await;

    let Ok(check_button) = client.find(Locator::Css(CHECK_TASK_BUTTON_SELECTOR)).await else {
        return CheckAttempt::ButtonMissing;
    };
    if check_button.click().await.is_err() {
        return CheckAttempt::ClickFailed;
    }
    println!("Clicked check, waiting for a result...");
    CheckAttempt::Result(wait_for_result(client).await)
}

/// Runs the whole solve-submit-check loop for the practice task currently
/// on screen, up to `config.max_attempts` times, logging every step. Unlike
/// the rest of the polling loop (one small DOM step per poll tick), this
/// runs to completion in a single call: each step here is inherently slow
/// (a network round trip to the model, or the site's own check), so
/// there's no benefit to splitting it across poll ticks the way a button
/// click is.
pub async fn try_solve(client: &Client, config: &PracticeConfig) {
    println!("Practice task detected — extracting task and files...");

    let task = match wait_for_task_description(client).await {
        Some(text) => text,
        None => {
            eprintln!(
                "Task description stayed empty for {}s after navigating here — stopping \
                 rather than solving against stale or missing content.",
                TASK_READY_TIMEOUT.as_secs()
            );
            return;
        }
    };

    // Some tasks (confirmed live: an end-of-topic checkpoint whose text
    // said outright "there's nothing to do here, just move on") already
    // pass with their starter content untouched. Rather than recognizing
    // that from the task's wording — brittle, since the exact phrasing
    // isn't guaranteed to repeat — just try the check as-is first, for
    // every task: it's the same signal the site itself uses to decide
    // pass/fail, so it can't be fooled by phrasing, and it's also a
    // freebie whenever the model would otherwise get a task right on the
    // first try anyway. Only downside is one extra check-click's worth of
    // latency on tasks that do need solving, which the site itself
    // usually resolves in well under `RESULT_POLL_TIMEOUT`.
    println!("Trying the check as-is first, in case this task needs no changes...");
    let mut previous_error: Option<String> = match click_check_and_wait(client).await {
        CheckAttempt::Result(CheckOutcome::Success) => {
            println!("Practice task passed without any changes.");
            return;
        }
        CheckAttempt::Result(CheckOutcome::Failure(error_text)) => Some(error_text),
        CheckAttempt::Result(CheckOutcome::Unknown) => {
            eprintln!(
                "No result notification appeared within {}s — stopping rather than guessing \
                 whether this passed.",
                RESULT_POLL_TIMEOUT.as_secs()
            );
            return;
        }
        CheckAttempt::ButtonMissing => {
            eprintln!("Check button disappeared — stopping.");
            return;
        }
        CheckAttempt::ClickFailed => {
            eprintln!("Couldn't click the check button — stopping.");
            return;
        }
    };

    let hint = get_hint(client).await;
    if let Some(hint) = &hint {
        println!("Hint: {hint}");
    }

    let mut files = match get_files(client).await {
        Ok(files) if !files.is_empty() => files,
        Ok(_) => {
            eprintln!("No editable files found in Monaco — skipping this task.");
            return;
        }
        Err(e) => {
            eprintln!("Couldn't read editor contents: {e}");
            return;
        }
    };

    // Confirmed live: at `BASE_TEMPERATURE`'s near-greedy sampling, the
    // model can reproduce the exact same (wrong) response across several
    // attempts in a row even though the corrective feedback in the prompt
    // was genuinely different each time (our own local correction, then
    // two different real site error messages) — a deterministic local
    // optimum, not a considered best answer. Once that's detected (the raw
    // response text is byte-identical to the previous attempt's),
    // remaining attempts switch to `STUCK_TEMPERATURE` so the model has an
    // actual chance at a different completion, instead of certainly
    // repeating the same failing one for every attempt left.
    const BASE_TEMPERATURE: f32 = 0.2;
    const STUCK_TEMPERATURE: f32 = 0.6;
    let mut temperature = BASE_TEMPERATURE;
    let mut last_response: Option<String> = None;

    for attempt in 1..=config.max_attempts {
        println!(
            "Attempt {attempt}/{}: asking {} for a solution...",
            config.max_attempts, config.model
        );

        let prompt = build_prompt(
            &task,
            hint.as_deref(),
            &files,
            previous_error.as_deref(),
            last_response.as_deref(),
        );
        let response = match ollama::complete(&config.model, &prompt, temperature).await {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Solver call failed: {e}");
                return;
            }
        };
        println!("--- model response ---\n{response}\n--- end response ---");

        if last_response.as_deref() == Some(response.as_str()) {
            eprintln!(
                "Attempt {attempt}: model repeated its previous answer verbatim despite \
                 different feedback — raising sampling temperature for the remaining attempts."
            );
            temperature = STUCK_TEMPERATURE;
        }
        last_response = Some(response.clone());

        let mut solution = parse_solution(&response);
        if solution.is_empty() {
            solution = guess_single_file_solution(&response, &files);
        }
        if solution.is_empty() {
            eprintln!("Couldn't parse a solution out of the model's response — giving up.");
            return;
        }

        for (filename, content) in &solution {
            match write_file(client, filename, content).await {
                Ok(true) => {
                    println!("Inserted solution into: {filename}");
                    if let Some(existing) = files.iter_mut().find(|(name, _)| name == filename) {
                        existing.1 = content.clone();
                    }
                }
                Ok(false) => eprintln!("No editor found for '{filename}' — skipped."),
                Err(e) => eprintln!("Couldn't write to '{filename}': {e}"),
            }
        }

        match click_check_and_wait(client).await {
            CheckAttempt::Result(CheckOutcome::Success) => {
                println!("Practice task passed.");
                return;
            }
            CheckAttempt::Result(CheckOutcome::Failure(error_text)) => {
                println!("Attempt {attempt} failed: {error_text}");
                previous_error = Some(error_text);
            }
            CheckAttempt::Result(CheckOutcome::Unknown) => {
                eprintln!(
                    "No result notification appeared within {}s — stopping rather than \
                     guessing whether this passed.",
                    RESULT_POLL_TIMEOUT.as_secs()
                );
                return;
            }
            CheckAttempt::ButtonMissing => {
                eprintln!("Check button disappeared — stopping.");
                return;
            }
            CheckAttempt::ClickFailed => {
                eprintln!("Couldn't click the check button — stopping.");
                return;
            }
        }
    }

    println!(
        "Giving up after {} attempts — leaving this task for you to solve manually.",
        config.max_attempts
    );
}

/// Polls `.task-description` until it has non-empty text, up to
/// `TASK_READY_TIMEOUT` — see its doc comment for why a fixed delay isn't
/// enough. Returns `None` if it never populates in time.
async fn wait_for_task_description(client: &Client) -> Option<String> {
    let deadline = std::time::Instant::now() + TASK_READY_TIMEOUT;
    loop {
        if let Ok(el) = client.find(Locator::Css(TASK_DESCRIPTION_SELECTOR)).await
            && let Ok(text) = el.text().await
            && !text.trim().is_empty()
        {
            return Some(text);
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        sleep(TASK_READY_POLL_INTERVAL).await;
    }
}

/// Opens the task's hint (if it has one) and returns its text. Best-effort
/// and entirely optional — a hint is a quality-of-solution improvement,
/// not something worth aborting the solve over, so any failure here (no
/// hint button, it doesn't open in time, ...) just means solving proceeds
/// without one.
async fn get_hint(client: &Client) -> Option<String> {
    let button = client.find(Locator::Css(HINT_BUTTON_SELECTOR)).await.ok()?;

    let already_open = button
        .attr("class")
        .await
        .ok()
        .flatten()
        .is_some_and(|classes| classes.split_whitespace().any(|c| c == HINT_BUTTON_ACTIVE_CLASS));

    if !already_open && button.click().await.is_err() {
        return None;
    }

    let notification = client
        .wait()
        .at_most(HINT_APPEAR_TIMEOUT)
        .for_element(Locator::Css(HINT_NOTIFICATION_SELECTOR))
        .await
        .ok()?;
    let content = notification
        .find(Locator::Css(NOTIFICATION_CONTENT_SELECTOR))
        .await
        .ok()?;
    let text = content.text().await.ok()?;

    // Close it back up rather than leaving it open: `NOTIFICATION_SELECTOR`
    // deliberately doesn't match `_type_hint`, so an open hint wouldn't
    // otherwise interfere with `dismiss_notification`/`wait_for_result` —
    // but there's no reason to leave it sitting on screen for the rest of
    // the solve loop either.
    if let Ok(close_button) = notification.find(Locator::Css(NOTIFICATION_CLOSE_SELECTOR)).await {
        let _ = close_button.click().await;
    }

    Some(text)
}

/// Reads every Monaco model's URI and current content.
async fn get_files(client: &Client) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let result = client
        .execute(
            "return monaco.editor.getModels().map(m => ({ uri: m.uri.toString(), value: m.getValue() }));",
            Vec::new(),
        )
        .await?;

    let array = result
        .as_array()
        .ok_or("expected an array of editor models")?;

    Ok(array
        .iter()
        .filter_map(|entry| {
            let uri = entry.get("uri")?.as_str()?;
            let value = entry.get("value")?.as_str()?;
            // Model URIs look like "file:///script.js" — strip the scheme
            // and leading slashes down to a plain filename, which is also
            // what we ask the model to use in its response.
            let name = uri.rsplit('/').next().unwrap_or(uri);
            Some((name.to_string(), value.to_string()))
        })
        .collect())
}

/// The extension-to-language mapping Monaco/the trainer UI uses for its
/// per-file editor containers (`.trainer-editor__code-editor_lang_<lang>`).
fn monaco_language(filename: &str) -> &str {
    match filename.rsplit('.').next().unwrap_or("") {
        "js" => "javascript",
        "html" | "htm" => "html",
        "css" => "css",
        "ts" => "typescript",
        "json" => "json",
        other => other,
    }
}

/// `Cmd+A` on macOS, `Ctrl+A` everywhere else.
fn select_all_combo() -> String {
    let modifier = if cfg!(target_os = "macos") {
        Key::Meta
    } else {
        Key::Control
    };
    modifier + "a"
}

/// How long to wait for an editor to become visible/interactable before
/// giving up on it. Editors mount asynchronously right after navigating to
/// a new practice task — confirmed live: an immediate `client.find` missed
/// one two attempts in a row for a page that had just loaded. Separately,
/// also confirmed live: a task with multiple files doesn't always open
/// with the file we want to write to as the active tab, and the editor
/// pane's `trainer-editor__code-editor_opened` class is *not* a reliable
/// signal of which one is actually visible — caught it live sitting on a
/// pane with a genuine zero-size `.view-lines` while the *other* pane
/// (without `_opened`) had the real, correctly-sized, visible one. Real
/// rendered visibility (`Element::is_displayed`) is what's checked
/// instead, in `wait_for_visible` below.
const EDITOR_APPEAR_TIMEOUT: Duration = Duration::from_secs(5);
const EDITOR_APPEAR_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Clicks the tab labeled with `filename` in the editor's tab bar, if one
/// exists — best-effort, since a single-file task has no tab bar to click
/// at all, and that's fine. Targets the tab's parent `<section>`, not the
/// `.tab__text` label itself — confirmed live that WebDriver refuses to
/// click the label directly ("element click intercepted", it's obscured
/// by a sibling hint-icon wrapper), while its parent is a normal,
/// unobstructed click target.
async fn switch_to_tab(client: &Client, filename: &str) {
    let xpath = format!(
        "//*[contains(@class, 'tab__text') and normalize-space(text())='{filename}']/parent::*"
    );
    if let Ok(tab) = client.find(Locator::XPath(&xpath)).await {
        let _ = tab.click().await;
    }
}

/// Polls `element.is_displayed()` for up to `EDITOR_APPEAR_TIMEOUT`.
async fn wait_for_visible(element: &Element) -> bool {
    let deadline = std::time::Instant::now() + EDITOR_APPEAR_TIMEOUT;
    loop {
        if element.is_displayed().await.unwrap_or(false) {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        sleep(EDITOR_APPEAR_POLL_INTERVAL).await;
    }
}

/// Monaco's own auto-closing-bracket/quote and auto-indent-on-Enter
/// features are meant for a human typing character by character, not for
/// pasting in an already fully-formed, already-indented multi-line
/// solution — confirmed live: typing content like `let x = [\n  'a',\n
/// 'b',\n];` as real keystrokes left a stray extra `]` at the very end
/// (the bracket Monaco auto-inserted the instant `[` was typed, then
/// pushed further down by every subsequent newline instead of being
/// reused) plus compounding indentation on every line. Disabling these
/// options is a pure editor-behavior setting, not a content mutation, so
/// it doesn't touch the "must be a real keystroke event, not
/// `setValue()`" requirement above — `send_keys` below still does the
/// actual writing. Applied to every editor instance on the page (there's
/// one per open file tab) since it's harmless on ones we're not currently
/// writing to.
const DISABLE_AUTO_PAIRING_SCRIPT: &str = "\
    monaco.editor.getEditors().forEach(e => e.updateOptions({\
        autoClosingBrackets: 'never',\
        autoClosingQuotes: 'never',\
        autoSurround: 'never',\
        autoIndent: 'none'\
    }));";

/// Replaces a file's content by focusing its editor and typing real
/// keystrokes — see the module doc comment for why this can't be a
/// `model.setValue()` call. Returns whether a matching editor was found.
async fn write_file(client: &Client, filename: &str, content: &str) -> Result<bool, Box<dyn Error>> {
    switch_to_tab(client, filename).await;

    let lang = monaco_language(filename);
    let view_lines_selector = format!(".trainer-editor__code-editor_lang_{lang} .view-lines");
    let input_selector = format!(".trainer-editor__code-editor_lang_{lang} textarea.inputarea");

    // Clicking the visible rendered lines is what actually focuses
    // Monaco's hidden textarea (Monaco's own click handler does that
    // internally) — `send_keys` below targets the textarea itself, since
    // it refuses non-interactive elements like `.view-lines`.
    let Ok(view_lines) = client.find(Locator::Css(&view_lines_selector)).await else {
        return Ok(false);
    };
    if !wait_for_visible(&view_lines).await {
        return Ok(false);
    }
    if view_lines.click().await.is_err() {
        return Ok(false);
    }

    let _ = client.execute(DISABLE_AUTO_PAIRING_SCRIPT, Vec::new()).await;

    let Ok(input) = client.find(Locator::Css(&input_selector)).await else {
        return Ok(false);
    };
    input.send_keys(&select_all_combo()).await?;
    input.send_keys(content).await?;

    Ok(true)
}

/// Best-effort: closes a result notification if one is currently visible.
/// Ignored if there isn't one or the close click fails — this is just
/// hygiene before the next check, not something worth failing the attempt
/// over.
async fn dismiss_notification(client: &Client) {
    if let Ok(notification) = client.find(Locator::Css(NOTIFICATION_SELECTOR)).await
        && let Ok(close_button) = notification.find(Locator::Css(NOTIFICATION_CLOSE_SELECTOR)).await
    {
        let _ = close_button.click().await;
    }
}

enum CheckOutcome {
    Success,
    Failure(String),
    /// Neither a success nor an error notification appeared within
    /// `RESULT_POLL_TIMEOUT` — genuinely ambiguous, not silently treated
    /// as either outcome.
    Unknown,
}

/// Polls for a result notification for up to `RESULT_POLL_TIMEOUT`,
/// distinguishing success from failure by the notification's own class
/// (`notification_type_success` vs `notification_type_error`) — confirmed
/// live rather than inferred, see the module doc comment.
async fn wait_for_result(client: &Client) -> CheckOutcome {
    let deadline = std::time::Instant::now() + RESULT_POLL_TIMEOUT;
    loop {
        if let Ok(notification) = client.find(Locator::Css(NOTIFICATION_SELECTOR)).await
            && let Ok(class_attr) = notification.attr("class").await
        {
            let classes = class_attr.unwrap_or_default();
            if classes.contains(NOTIFICATION_SUCCESS_CLASS) {
                return CheckOutcome::Success;
            }
            if classes.contains(NOTIFICATION_ERROR_CLASS)
                && let Ok(content) = notification
                    .find(Locator::Css(NOTIFICATION_CONTENT_SELECTOR))
                    .await
                && let Ok(text) = content.text().await
            {
                return CheckOutcome::Failure(text);
            }
        }

        if std::time::Instant::now() >= deadline {
            return CheckOutcome::Unknown;
        }
        sleep(RESULT_POLL_INTERVAL).await;
    }
}

fn build_prompt(
    task: &str,
    hint: Option<&str>,
    files: &[(String, String)],
    previous_error: Option<&str>,
    previous_response: Option<&str>,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "Ты решаешь практическое задание на образовательной платформе для веб-разработки.\n\n\
         Пиши минимально необходимый код, который точно выполняет условие — без лишних \
         промежуточных переменных, преобразований или форматирования вывода, которых условие \
         не требует явно. Проверка обычно ищет конкретный результат, а не «красивое» решение.\n\n\
         Текущее содержимое файлов ниже уже отражает правильно решённые предыдущие шаги этого же \
         задания — не переписывай, не удаляй и не заменяй уже существующие строки (например, не \
         меняй начальное значение переменной и не превращай существующие выражения в другие, даже \
         эквивалентные по смыслу). Если условие не требует явно изменить существующую строку — \
         просто допиши недостающий код в конец, оставив всё остальное точно как было.\n\n",
    );
    prompt.push_str("Условие задания:\n");
    prompt.push_str(task);
    prompt.push('\n');
    if let Some(hint) = hint {
        prompt.push_str(
            "\nПодсказка к заданию (следуй ей буквально — если в ней уже дано готовое \
             выражение или код, вставь его как есть, не изменяя и не расширяя):\n",
        );
        prompt.push_str(hint);
        prompt.push('\n');
    }
    prompt.push_str("\nТекущее содержимое файлов:\n\n");
    for (name, content) in files {
        prompt.push_str(&format!("FILE: {name}\n```\n{content}\n```\n\n"));
    }
    if let Some(error) = previous_error {
        prompt.push_str("Предыдущая попытка не прошла проверку. Текст ошибки:\n");
        prompt.push_str(error);
        prompt.push_str("\n\n");
        if let Some(prev_response) = previous_response {
            prompt.push_str(
                "Вот твой предыдущий ответ целиком, который и привёл к этой ошибке — НЕ \
                 повторяй его снова в том же виде, предложи другой вариант, который её \
                 устраняет:\n",
            );
            prompt.push_str(prev_response);
            prompt.push_str("\n\n");
        }
    }
    prompt.push_str(
        "Верни только те файлы, которые нужно изменить, в точности в следующем формате, \
         без каких-либо пояснений до, после или между файлами:\n\n\
         FILE: <имя файла>\n\
         ```\n\
         <полное новое содержимое файла>\n\
         ```\n\n\
         Можно вернуть несколько файлов подряд в этом же формате.",
    );
    prompt
}

/// Splits a line like "```FILE: script.js" (confirmed live from
/// `qwen2.5-coder:3b`: the opening fence glued directly onto the `FILE:`
/// marker instead of sitting on its own line) into two proper lines —
/// `FILE: script.js` followed by a fresh opening fence — so the rest of
/// `parse_solution`/`extract_fenced_code` sees the same shape it always
/// expects, instead of needing a separate parsing path for this quirk.
fn normalize_glued_fence_file_markers(response: &str) -> String {
    let mut out = String::with_capacity(response.len() + 8);
    for line in response.lines() {
        if let Some(name) = line.trim_start().strip_prefix("```").and_then(|rest| rest.strip_prefix("FILE:")) {
            out.push_str("FILE:");
            out.push_str(name);
            out.push('\n');
            out.push_str("```\n");
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Parses the model's response for `FILE: <name>` markers each followed by
/// a fenced code block, matching the format requested in `build_prompt`.
/// Tolerates a quirk confirmed live with `qwen2.5-coder:3b` — across
/// otherwise-identical prompts, it occasionally emits a spurious empty
/// ` ``` ``` ` pair right after the filename before the real code block —
/// by not assuming the first fence pair found is the real one (see
/// `extract_fenced_code`).
fn parse_solution(response: &str) -> Vec<(String, String)> {
    let response = normalize_glued_fence_file_markers(response);
    let lines: Vec<&str> = response.lines().collect();
    let mut files = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let Some(filename) = lines[i].trim().strip_prefix("FILE:").map(str::trim) else {
            i += 1;
            continue;
        };

        let block_end = lines[i + 1..]
            .iter()
            .position(|l| l.trim().strip_prefix("FILE:").is_some())
            .map(|rel| i + 1 + rel)
            .unwrap_or(lines.len());

        if let Some(code) = extract_fenced_code(&lines[i + 1..block_end]) {
            files.push((filename.to_string(), code));
        }
        i = block_end;
    }

    files
}

/// Fallback for when the model skips the `FILE:` markers entirely and just
/// returns a bare code block — a reasonable simplification for the model
/// to make when there's obviously only one file to fill in. Only applies
/// when exactly one of the known files is currently empty; otherwise
/// there's no safe way to guess which file a bare block belongs to.
fn guess_single_file_solution(response: &str, files: &[(String, String)]) -> Vec<(String, String)> {
    let mut empty_files = files.iter().filter(|(_, content)| content.trim().is_empty());
    let (Some((name, _)), None) = (empty_files.next(), empty_files.next()) else {
        return Vec::new();
    };

    let lines: Vec<&str> = response.lines().collect();
    match extract_fenced_code(&lines) {
        Some(code) => vec![(name.clone(), code)],
        None => Vec::new(),
    }
}

/// Every gap between two consecutive fenced-code-block markers (bare
/// ` ``` ` lines) in `lines` is a candidate; the longest non-empty one
/// wins. This — rather than assuming the first `` ``` ``...``` `` `` pair
/// found is the intended block — is what lets `parse_solution` recover
/// from a spurious empty fence pair placed before the real one.
fn extract_fenced_code(lines: &[&str]) -> Option<String> {
    let fence_positions: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim_start().starts_with("```"))
        .map(|(i, _)| i)
        .collect();

    fence_positions
        .windows(2)
        .map(|w| lines[w[0] + 1..w[1]].join("\n"))
        .filter(|s| !s.trim().is_empty())
        .max_by_key(String::len)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured live from qwen2.5-coder:3b against an identical prompt,
    // three requests in a row — the model isn't consistent about format
    // even then, which is exactly why `parse_solution` needs to tolerate
    // more than the one "clean" shape.
    const WELL_FORMED: &str = "FILE: script.js\n```\ndocument.write('В чём сила?');\n```";
    const SPURIOUS_EMPTY_FENCE: &str =
        "```\nFILE: script.js\n```\n```\ndocument.write('В чём сила?');\n```";

    #[test]
    fn parses_well_formed_response() {
        let files = parse_solution(WELL_FORMED);
        assert_eq!(
            files,
            vec![(
                "script.js".to_string(),
                "document.write('В чём сила?');".to_string()
            )]
        );
    }

    #[test]
    fn recovers_from_spurious_empty_fence_before_the_real_one() {
        let files = parse_solution(SPURIOUS_EMPTY_FENCE);
        assert_eq!(
            files,
            vec![(
                "script.js".to_string(),
                "document.write('В чём сила?');".to_string()
            )]
        );
    }

    #[test]
    fn guesses_the_single_empty_file_when_no_file_marker_present() {
        let response = "```\ndocument.write('В чём сила?');\n```";
        let files = vec![
            ("index.html".to_string(), "<html></html>".to_string()),
            ("script.js".to_string(), String::new()),
        ];
        let solution = guess_single_file_solution(response, &files);
        assert_eq!(
            solution,
            vec![(
                "script.js".to_string(),
                "document.write('В чём сила?');".to_string()
            )]
        );
    }

    #[test]
    fn does_not_guess_when_multiple_files_are_empty() {
        let response = "```\ndocument.write('В чём сила?');\n```";
        let files = vec![
            ("index.html".to_string(), String::new()),
            ("script.js".to_string(), String::new()),
        ];
        assert!(guess_single_file_solution(response, &files).is_empty());
    }

    #[test]
    fn recovers_from_fence_glued_directly_to_file_marker() {
        // Captured live from qwen2.5-coder:3b: "```FILE: script.js" on one
        // line instead of "```" and "FILE: script.js" on separate lines.
        let response = "```FILE: script.js\nlet time = 0;\n```";
        let files = parse_solution(response);
        assert_eq!(files, vec![("script.js".to_string(), "let time = 0;".to_string())]);
    }

    #[test]
    fn parse_solution_ignores_surrounding_prose() {
        let response = format!("Конечно, вот решение:\n\n{WELL_FORMED}\n\nНадеюсь, это поможет!");
        let files = parse_solution(&response);
        assert_eq!(
            files,
            vec![(
                "script.js".to_string(),
                "document.write('В чём сила?');".to_string()
            )]
        );
    }

    #[test]
    fn build_prompt_includes_hint_when_present() {
        let files = vec![("script.js".to_string(), String::new())];
        let prompt = build_prompt("Условие.", Some("Используй console.log."), &files, None, None);
        assert!(prompt.contains("Подсказка"));
        assert!(prompt.contains("Используй console.log."));
    }

    #[test]
    fn build_prompt_omits_hint_section_when_absent() {
        let files = vec![("script.js".to_string(), String::new())];
        let prompt = build_prompt("Условие.", None, &files, None, None);
        assert!(!prompt.contains("Подсказка"));
    }

    #[test]
    fn build_prompt_includes_previous_response_alongside_error() {
        let files = vec![("script.js".to_string(), String::new())];
        let prompt = build_prompt(
            "Условие.",
            None,
            &files,
            Some("Неверное значение переменной time."),
            Some("FILE: script.js\n```\nlet time = 0;\n```"),
        );
        assert!(prompt.contains("НЕ повторяй его"));
        assert!(prompt.contains("let time = 0;"));
    }

    #[test]
    fn build_prompt_omits_previous_response_without_an_error() {
        let files = vec![("script.js".to_string(), String::new())];
        // No `previous_error` means this is attempt 1 — nothing to avoid
        // repeating yet, so `previous_response` (which would only be set
        // from a prior attempt) shouldn't surface even if passed.
        let prompt = build_prompt("Условие.", None, &files, None, Some("some earlier text"));
        assert!(!prompt.contains("some earlier text"));
    }
}
