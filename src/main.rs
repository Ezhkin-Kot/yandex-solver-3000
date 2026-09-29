mod ollama;
mod practice;

use std::error::Error;
use std::process::Stdio;
use std::time::Duration;

use fantoccini::elements::Element;
use fantoccini::error::NewSessionError;
use fantoccini::{Client, ClientBuilder, Locator};
use hyper_util::client::legacy::connect::HttpConnector;
use serde_json::{Map, Value as Json};
use tokio::process::{Child, Command};
use tokio::time::sleep;

use practice::PracticeConfig;

/// Every block in Practicum's chat-style theory viewer is wrapped in a
/// `<div class="... theory-viewer__block_type_<kind> ...">` — `dialog` and
/// `markdown` for plain content, `action-button` for the "reveal next
/// bubble" button, `quiz-select` for a quiz, and presumably others we
/// haven't seen (code tasks, etc). The "go to next lesson" control isn't
/// one of these blocks — it's a sibling that appears once the lesson's
/// blocks are exhausted.
///
/// Querying both together lets us look at whichever is *last in DOM
/// order* to decide what's actually going on right now: a comma-separated
/// CSS selector list returns matches in document order regardless of
/// which part of the list matched, so this naturally prefers the
/// next-lesson button (or a freshly-added block) over an older,
/// already-resolved one still sitting in the DOM.
const THEORY_BLOCK_SELECTOR: &str =
    "[class*='theory-viewer__block_type_'], [data-test-id='next-lesson-control-button']";

const NEXT_LESSON_BUTTON_TEST_ID: &str = "next-lesson-control-button";
const CONTENT_EXPANDER_SELECTOR: &str = ".content-expander__button";

/// `theory-viewer__block_type_quiz-select` covers both single-answer
/// (radio) and multi-answer (checkbox) quizzes — they share the same
/// block/option markup and only differ in the `<input>`'s `type`, which
/// YOLO mode doesn't need to distinguish: clicking the option's `<label>`
/// (not the visually-hidden `<input>` itself, which WebDriver may refuse
/// to click as non-interactable) toggles it either way.
const QUIZ_SELECT_BLOCK_KIND: &str = "quiz-select";
const QUIZ_OPTION_LABEL_SELECTOR: &str = ".quiz-form-choice__option-label";
const QUIZ_SUBMIT_SELECTOR: &str = ".quiz__submit";

/// Text-based fallback for lesson types without the classes above, checked
/// via a single XPath (also keeps document order) against every <button>.
const TEXT_FALLBACK_XPATH: &str = "//button[\
    contains(., 'Дальше') or \
    contains(., 'Далее') or \
    contains(., 'Продолжить') or \
    contains(., 'Следующий шаг') or \
    contains(., 'Следующий урок') or \
    contains(., 'Перейти к следующему шагу')\
]";

const POLL_INTERVAL: Duration = Duration::from_millis(500);
const POST_CLICK_SETTLE_DELAY: Duration = Duration::from_millis(1000);

const GECKODRIVER_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const GECKODRIVER_CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(300);

/// Both Zen and Firefox are Gecko browsers geckodriver drives the same
/// way (same `-profile`/`-marionette` launch args, same WebDriver
/// protocol) — the only per-browser bits are the default binary path and
/// which profile directory to use, so each gets its own profile rather
/// than sharing one meant for the other browser's install.
#[derive(Clone, Copy)]
enum Browser {
    Zen,
    Firefox,
}

impl Browser {
    fn parse(name: &str) -> Result<Self, Box<dyn Error>> {
        match name.to_ascii_lowercase().as_str() {
            "zen" => Ok(Browser::Zen),
            "firefox" => Ok(Browser::Firefox),
            other => Err(format!("unknown BROWSER '{other}'; expected 'zen' or 'firefox'").into()),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Browser::Zen => "Zen",
            Browser::Firefox => "Firefox",
        }
    }

    /// macOS app bundles live at a fixed, well-known path, so we can point
    /// straight at the binary inside one. Linux installs don't have an
    /// equivalent standard location (`/usr/bin`, `/opt/...`, a Flatpak
    /// wrapper, an AppImage anywhere on disk, ...), so there we fall back
    /// to the bare command name and let `$PATH` resolve it — same as
    /// `geckodriver` itself already does below.
    fn default_binary_path(self) -> &'static str {
        if cfg!(target_os = "macos") {
            match self {
                Browser::Zen => "/Applications/Zen.app/Contents/MacOS/zen",
                Browser::Firefox => "/Applications/Firefox.app/Contents/MacOS/firefox",
            }
        } else {
            match self {
                Browser::Zen => "zen",
                Browser::Firefox => "firefox",
            }
        }
    }

    fn default_profile_dir_name(self) -> &'static str {
        match self {
            Browser::Zen => ".zen-bot-profile",
            Browser::Firefox => ".firefox-bot-profile",
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load config from a .env file in the working directory, if present,
    // without overriding variables already set in the environment. Missing
    // file is fine — all of these have defaults anyway.
    let _ = dotenvy::dotenv();

    let port = std::env::var("GECKODRIVER_PORT").unwrap_or_else(|_| "4445".into());
    let yolo_mode = env_flag("YOLO_MODE");

    let practice_config = if env_flag("AI_SOLVE_MODE") {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5-coder:3b".into());
        let max_attempts = std::env::var("MAX_SOLVE_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        Some(PracticeConfig { model, max_attempts })
    } else {
        None
    };

    let browser = match std::env::var("BROWSER") {
        Ok(name) => Browser::parse(&name)?,
        Err(_) => Browser::Zen,
    };
    let binary = std::env::var("BROWSER_BINARY_PATH")
        .unwrap_or_else(|_| browser.default_binary_path().to_string());
    let profile_dir = std::env::var("BROWSER_PROFILE_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME is not set");
        format!("{home}/{}", browser.default_profile_dir_name())
    });

    if !binary_exists(&binary) {
        return Err(format!(
            "{} binary '{binary}' not found; set BROWSER_BINARY_PATH to its full path, \
             or BROWSER=zen / BROWSER=firefox to pick the other default",
            browser.label()
        )
        .into());
    }
    std::fs::create_dir_all(&profile_dir)?;

    // Same lifecycle as geckodriver below: if AI_SOLVE_MODE is on and
    // nothing is already listening, we start Ollama ourselves and stop it
    // on exit — no standing background service required for this project.
    let mut ollama_process = None;
    if practice_config.is_some() {
        ollama_process = ollama::ensure_running().await?;
    }

    println!("Starting geckodriver on port {port}...");
    let mut geckodriver = spawn_geckodriver(&port)?;

    let result = run(
        browser,
        &port,
        &binary,
        &profile_dir,
        yolo_mode,
        practice_config,
    )
    .await;

    // Best-effort cleanup: don't leave a geckodriver process (and the
    // browser it spawned), or an ollama serve we started ourselves (and
    // *its* model-runner child process — see `ollama::stop`), running
    // after we exit, whether that's because of an error or a clean
    // Ctrl+C.
    let _ = geckodriver.kill().await;
    if let Some(child) = ollama_process {
        ollama::stop(child).await;
    }

    result
}

/// Parses a boolean-ish env var (`1`/`true`/`yes`, case-insensitive);
/// unset or anything else means off.
fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

/// An absolute/relative path (containing a `/`) is checked directly; a
/// bare command name (the Linux default, see `default_binary_path`) is
/// instead looked up in `$PATH`, the same way the OS would resolve it
/// when we later hand it to geckodriver.
fn binary_exists(binary: &str) -> bool {
    if binary.contains('/') {
        return std::path::Path::new(binary).exists();
    }

    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(binary).is_file()))
}

/// Sends SIGTERM (not SIGKILL — gives the browser a chance to shut down
/// normally and flush its profile) to any process whose command line
/// contains `profile_dir`. Matching on the profile path rather than a
/// process name means this can only ever hit a process that was launched
/// with `-profile <profile_dir>` — i.e. an instance of *this bot's own*
/// dedicated profile, never the user's regular browser session, which
/// runs against a different profile path entirely.
fn kill_stale_browser_process(profile_dir: &str) {
    let _ = std::process::Command::new("pkill")
        .args(["-f", profile_dir])
        .status();
}

/// Belt-and-suspenders alongside `kill_stale_browser_process`: even once
/// the process is gone, Gecko's profile-lock artifacts
/// (`lock`/`.parentlock`) can occasionally outlive it. Removing files that
/// don't exist is a no-op, so this is safe to call unconditionally.
fn clear_stale_profile_lock(profile_dir: &str) {
    for name in ["lock", ".parentlock"] {
        let _ = std::fs::remove_file(std::path::Path::new(profile_dir).join(name));
    }
}

fn spawn_geckodriver(port: &str) -> Result<Child, Box<dyn Error>> {
    Command::new("geckodriver")
        .arg("--port")
        .arg(port)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            format!("failed to start geckodriver ({e}); is it installed and on PATH?").into()
        })
}

async fn run(
    browser: Browser,
    port: &str,
    binary: &str,
    profile_dir: &str,
    yolo_mode: bool,
    practice_config: Option<PracticeConfig>,
) -> Result<(), Box<dyn Error>> {
    let client = match connect(port, binary, profile_dir).await {
        Ok(client) => client,
        Err(first_err) => {
            // A previous run that didn't shut down cleanly (crash, `kill
            // -9`, a closed terminal, ...) can leave the browser process —
            // or just its profile lock file — behind: our own cleanup on
            // exit only reaches the geckodriver process, not the browser
            // it spawns as its own child. That shows up here as a failed
            // connection (the browser refuses to start, showing its own
            // "already running" dialog). We only ever retry once, and
            // only after failing normally first — never preemptively —
            // and only against `profile_dir`, which is dedicated to this
            // bot and never the user's real browser profile.
            eprintln!(
                "Couldn't start {}: {first_err}\n\
                 This usually means a previous run left its browser process (or just its \
                 profile lock) behind. Closing whatever is using {profile_dir} and retrying \
                 once — this only ever touches the bot's own dedicated profile, never your \
                 regular browser session.",
                browser.label()
            );
            kill_stale_browser_process(profile_dir);
            sleep(Duration::from_millis(500)).await;
            clear_stale_profile_lock(profile_dir);
            connect(port, binary, profile_dir).await?
        }
    };
    let label = browser.label();
    println!("Connected — {label} window is up.");
    if yolo_mode {
        println!("YOLO mode is on — quizzes will be answered by picking a random option.");
    }

    client.goto("https://practicum.yandex.ru").await?;
    println!(
        "Log in and open your lesson in the {label} window that just opened. \
         The bot will start clicking automatically once it recognizes lesson content. \
         Press Ctrl+C here to stop."
    );

    let poll_loop = poll_and_click(&client, yolo_mode, practice_config.as_ref());
    tokio::select! {
        result = poll_loop => result?,
        _ = tokio::signal::ctrl_c() => {
            println!("\nStopping...");
        }
    }

    // Closing the session tells geckodriver to close the browser window
    // too, instead of leaving an orphaned Zen process behind.
    let _ = client.close().await;
    Ok(())
}

/// geckodriver's HTTP server takes a moment to come up after we spawn it,
/// so retry the initial connection instead of failing on the first miss.
///
/// Only retries on a *transport*-level failure (`Failed`/`FailedC`/`Lost`
/// — nothing answering on the port yet), which has no side effect and is
/// safe to hammer every `GECKODRIVER_CONNECT_RETRY_INTERVAL`. Any other
/// error means geckodriver's server *did* respond — i.e. it already
/// tried (and failed) to create a session, which means actually launching
/// a browser process. Retrying that in the same tight loop would mean up
/// to ~50 browser-launch attempts in `GECKODRIVER_CONNECT_TIMEOUT`, each
/// potentially popping its own window — so those are returned immediately
/// instead, leaving retry decisions to the caller (see the single,
/// deliberate retry in `run`, which cleans up first).
async fn connect(port: &str, binary: &str, profile_dir: &str) -> Result<Client, Box<dyn Error>> {
    let mut firefox_options = Map::new();
    firefox_options.insert("binary".to_string(), Json::String(binary.to_string()));
    firefox_options.insert(
        "args".to_string(),
        Json::Array(vec![
            Json::String("-profile".to_string()),
            Json::String(profile_dir.to_string()),
        ]),
    );

    let mut capabilities = Map::new();
    capabilities.insert(
        "moz:firefoxOptions".to_string(),
        Json::Object(firefox_options),
    );

    let mut builder = ClientBuilder::new(HttpConnector::new());
    builder.capabilities(capabilities);

    let webdriver_url = format!("http://localhost:{port}");
    let deadline = std::time::Instant::now() + GECKODRIVER_CONNECT_TIMEOUT;
    loop {
        match builder.connect(&webdriver_url).await {
            Ok(client) => return Ok(client),
            Err(
                e @ (NewSessionError::Failed(_)
                | NewSessionError::FailedC(_)
                | NewSessionError::Lost(_)),
            ) => {
                if std::time::Instant::now() >= deadline {
                    return Err(
                        format!("couldn't connect to geckodriver at {webdriver_url}: {e}").into(),
                    );
                }
                sleep(GECKODRIVER_CONNECT_RETRY_INTERVAL).await;
            }
            Err(e) => {
                return Err(
                    format!("couldn't connect to geckodriver at {webdriver_url}: {e}").into(),
                );
            }
        }
    }
}

enum Step {
    /// Clicked a button; the label describes what was clicked.
    Clicked(String),
    /// The current block needs a human (quiz, code task, ...); the label
    /// names its `theory-viewer__block_type_*` kind for logging.
    Blocked(String),
    /// Nothing recognizable on the page right now.
    Idle,
}

async fn poll_and_click(
    client: &Client,
    yolo_mode: bool,
    practice_config: Option<&PracticeConfig>,
) -> Result<(), Box<dyn Error>> {
    let mut last_url = client.current_url().await?.to_string();
    let mut last_blocked_reason: Option<String> = None;
    let mut attempted_practice_urls = std::collections::HashSet::new();

    loop {
        if let Ok(url) = client.current_url().await {
            let url = url.to_string();
            if url != last_url {
                println!("Navigated to: {url}");
                last_url = url;
                last_blocked_reason = None;
            }
        }

        if let Some(config) = practice_config
            && !attempted_practice_urls.contains(&last_url)
            && practice::is_practice_task(client).await
            && !practice::theory_popup_open(client).await
        {
            attempted_practice_urls.insert(last_url.clone());
            practice::try_solve(client, config).await;
            sleep(POST_CLICK_SETTLE_DELAY).await;
            continue;
        }
        // Some lessons show their theory content as a popup layered on
        // top of the editor — see `practice::THEORY_POPUP_SELECTOR`. The
        // editor is unusable while it's up, so the branch above is
        // skipped (note: this URL is *not* marked attempted while that's
        // the case) and control falls through here instead, which is
        // what the popup actually needs (reveal dialog, answer quizzes,
        // the same as any other theory content).

        match try_click_next_button(client, yolo_mode).await {
            Step::Clicked(label) => {
                println!("Clicked ({label}), waiting for page to settle...");
                last_blocked_reason = None;
                // give the SPA time to render the next lesson before we poll again
                sleep(POST_CLICK_SETTLE_DELAY).await;
            }
            Step::Blocked(reason) => {
                if last_blocked_reason.as_deref() != Some(reason.as_str()) {
                    println!(
                        "Waiting on a manual step ({reason}) — clicks paused until it's resolved."
                    );
                    last_blocked_reason = Some(reason);
                }
            }
            Step::Idle => {}
        }

        sleep(POLL_INTERVAL).await;
    }
}

/// Closes the theory popup once its content is fully consumed (see
/// `practice::THEORY_POPUP_SELECTOR`) — its label varies by context
/// ("Перейти к заданию" when a practice task follows, presumably other
/// text otherwise), but `data-test-id="theory-panel-close-button"` stays
/// the same regardless, so this is preferred over guessing at every label
/// via `TEXT_FALLBACK_XPATH`.
const THEORY_PANEL_CLOSE_BUTTON_SELECTOR: &str = "[data-test-id='theory-panel-close-button']";

/// Tries the theory-popup close button first — when it's present and
/// enabled, that's the platform's own signal that this popup is done and
/// safe to leave, which should win regardless of what block a stale
/// trailing element (a "markdown" paragraph, an old block type we
/// wouldn't otherwise vet as safe, ...) happens to still be. Only once
/// that's unavailable does it fall to the theory-viewer block logic, and
/// only if *that* comes back `Idle` does it fall further to
/// `TEXT_FALLBACK_XPATH`. Previously the text fallback only ran when
/// `THEORY_BLOCK_SELECTOR` matched *nothing at all*, which meant a
/// stale-but-still-present block could permanently block ever reaching
/// it — confirmed live with an already-answered quiz, and separately with
/// a trailing "markdown" block sitting in front of an already-clickable
/// close button.
async fn try_click_next_button(client: &Client, yolo_mode: bool) -> Step {
    match try_theory_panel_close(client).await {
        Step::Idle => {}
        step => return step,
    }
    match try_theory_block(client, yolo_mode).await {
        Step::Idle => try_text_fallback(client).await,
        step => step,
    }
}

async fn try_theory_panel_close(client: &Client) -> Step {
    let Ok(button) = client
        .find(Locator::Css(THEORY_PANEL_CLOSE_BUTTON_SELECTOR))
        .await
    else {
        return Step::Idle;
    };

    let is_disabled = button.attr("disabled").await.ok().flatten().is_some();
    if !is_disabled && button.click().await.is_ok() {
        Step::Clicked(THEORY_PANEL_CLOSE_BUTTON_SELECTOR.to_string())
    } else {
        Step::Idle
    }
}

/// Looks at the last `THEORY_BLOCK_SELECTOR` match to decide what to do:
/// click the next-lesson button, click the current reveal-dialog button,
/// or back off because the last block is something else (a quiz, a code
/// task, ...) that needs a human.
async fn try_theory_block(client: &Client, yolo_mode: bool) -> Step {
    let Ok(blocks) = client.find_all(Locator::Css(THEORY_BLOCK_SELECTOR)).await else {
        return Step::Idle;
    };

    let Some(last_block) = blocks.into_iter().last() else {
        return Step::Idle;
    };

    let is_next_lesson_button = last_block
        .attr("data-test-id")
        .await
        .ok()
        .flatten()
        .as_deref()
        == Some(NEXT_LESSON_BUTTON_TEST_ID);

    if is_next_lesson_button {
        return if last_block.click().await.is_ok() {
            Step::Clicked(NEXT_LESSON_BUTTON_TEST_ID.to_string())
        } else {
            Step::Idle
        };
    }

    let class_attr = last_block
        .attr("class")
        .await
        .ok()
        .flatten()
        .unwrap_or_default();

    match class_attr
        .split_whitespace()
        .find_map(|c| c.strip_prefix("theory-viewer__block_type_"))
    {
        Some("action-button") => {
            match last_block
                .find(Locator::Css(CONTENT_EXPANDER_SELECTOR))
                .await
            {
                Ok(button) if button.click().await.is_ok() => {
                    Step::Clicked(CONTENT_EXPANDER_SELECTOR.to_string())
                }
                _ => Step::Idle,
            }
        }
        Some(QUIZ_SELECT_BLOCK_KIND) if yolo_mode => try_yolo_quiz(&last_block).await,
        // Plain content, never actionable by itself — confirmed across
        // every page-example so far: "dialog"/"markdown" are chat text or
        // paragraphs, "vertical-layout" is a wrapper around other blocks
        // (never actionable directly; see the module doc comment on
        // `THEORY_BLOCK_SELECTOR`). Idle here (rather than Blocked) lets
        // the caller's cascade keep looking — e.g. a theory-popup close
        // button that's already active regardless of this trailing block.
        Some("dialog" | "markdown" | "vertical-layout") => Step::Idle,
        Some(other_kind) => Step::Blocked(other_kind.to_string()),
        None => Step::Idle,
    }
}

async fn try_text_fallback(client: &Client) -> Step {
    if let Ok(elements) = client.find_all(Locator::XPath(TEXT_FALLBACK_XPATH)).await
        && let Some(element) = elements.into_iter().last()
        && element.click().await.is_ok()
    {
        return Step::Clicked("text fallback".to_string());
    }

    Step::Idle
}

/// Answers a quiz block by picking a random option and submitting, split
/// across two poll cycles rather than done in one shot: submitting only
/// becomes possible once the app's own JS notices the selection and drops
/// `disabled` off `.quiz__submit`, which needs a round trip through the
/// page's own state update — trying to click it in the very same tick we
/// select an option would just find it still disabled.
async fn try_yolo_quiz(block: &Element) -> Step {
    // A quiz that's already been answered gets a `quiz_answered` class and
    // drops `.quiz__submit` from the DOM entirely, but its options (now
    // disabled) stay — confirmed live. Without this check, once answered
    // and with nothing new appended after it (e.g. inside a theory popup
    // that doesn't chain another block here), this quiz stays the "last"
    // theory block forever: `.quiz__submit` is never found, so every poll
    // tick falls into the "pick a random option" branch below and clicks
    // an already-disabled option — WebDriver doesn't refuse that click,
    // so it reports success and nothing ever changes, forever. Idle here
    // lets `try_click_next_button` fall through to the text-based
    // fallback instead, since something other than a theory-viewer block
    // must be what actually advances past this.
    let class_attr = block.attr("class").await.ok().flatten().unwrap_or_default();
    if class_attr.split_whitespace().any(|c| c == "quiz_answered") {
        return Step::Idle;
    }

    if let Ok(submit) = block.find(Locator::Css(QUIZ_SUBMIT_SELECTOR)).await {
        let is_disabled = submit.attr("disabled").await.ok().flatten().is_some();
        if !is_disabled {
            return if submit.click().await.is_ok() {
                Step::Clicked("yolo: submitted quiz answer".to_string())
            } else {
                Step::Idle
            };
        }
    }

    let Ok(options) = block
        .find_all(Locator::Css(QUIZ_OPTION_LABEL_SELECTOR))
        .await
    else {
        return Step::Idle;
    };
    if options.is_empty() {
        return Step::Idle;
    }

    let index = rand::random_range(..options.len());
    if options[index].click().await.is_ok() {
        Step::Clicked(format!(
            "yolo: picked random quiz option {}/{}",
            index + 1,
            options.len()
        ))
    } else {
        Step::Idle
    }
}
