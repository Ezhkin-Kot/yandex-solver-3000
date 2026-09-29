use std::error::Error;
use std::process::Stdio;
use std::time::Duration;

use fantoccini::elements::Element;
use fantoccini::{Client, ClientBuilder, Locator};
use hyper_util::client::legacy::connect::HttpConnector;
use serde_json::{Map, Value as Json};
use tokio::process::{Child, Command};
use tokio::time::sleep;

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
    let yolo_mode = std::env::var("YOLO_MODE")
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"));

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

    println!("Starting geckodriver on port {port}...");
    let mut geckodriver = spawn_geckodriver(&port)?;

    let result = run(browser, &port, &binary, &profile_dir, yolo_mode).await;

    // Best-effort cleanup: don't leave a geckodriver process (and the
    // browser it spawned) running after we exit, whether that's because
    // of an error or a clean Ctrl+C.
    let _ = geckodriver.kill().await;

    result
}

/// An absolute/relative path (containing a `/`) is checked directly; a
/// bare command name (the Linux default, see `default_binary_path`) is
/// instead looked up in `$PATH`, the same way the OS would resolve it
/// when we later hand it to geckodriver.
fn binary_exists(binary: &str) -> bool {
    if binary.contains('/') {
        return std::path::Path::new(binary).exists();
    }

    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| dir.join(binary).is_file())
    })
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
) -> Result<(), Box<dyn Error>> {
    let client = connect(port, binary, profile_dir).await?;
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

    let poll_loop = poll_and_click(&client, yolo_mode);
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
    capabilities.insert("moz:firefoxOptions".to_string(), Json::Object(firefox_options));

    let mut builder = ClientBuilder::new(HttpConnector::new());
    builder.capabilities(capabilities);

    let webdriver_url = format!("http://localhost:{port}");
    let deadline = std::time::Instant::now() + GECKODRIVER_CONNECT_TIMEOUT;
    loop {
        match builder.connect(&webdriver_url).await {
            Ok(client) => return Ok(client),
            Err(e) => {
                if std::time::Instant::now() >= deadline {
                    return Err(format!("couldn't connect to geckodriver at {webdriver_url}: {e}").into());
                }
                sleep(GECKODRIVER_CONNECT_RETRY_INTERVAL).await;
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

async fn poll_and_click(client: &Client, yolo_mode: bool) -> Result<(), Box<dyn Error>> {
    let mut last_url = client.current_url().await?.to_string();
    let mut last_blocked_reason: Option<String> = None;

    loop {
        if let Ok(url) = client.current_url().await {
            let url = url.to_string();
            if url != last_url {
                println!("Navigated to: {url}");
                last_url = url;
                last_blocked_reason = None;
            }
        }

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

/// Looks at the last `THEORY_BLOCK_SELECTOR` match to decide what to do:
/// click the next-lesson button, click the current reveal-dialog button,
/// or back off because the last block is something else (a quiz, a code
/// task, ...) that needs a human. Falls back to `TEXT_FALLBACK_XPATH` for
/// lesson types that don't use the theory-viewer block classes at all.
async fn try_click_next_button(client: &Client, yolo_mode: bool) -> Step {
    let Ok(blocks) = client.find_all(Locator::Css(THEORY_BLOCK_SELECTOR)).await else {
        return Step::Idle;
    };

    if let Some(last_block) = blocks.into_iter().last() {
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

        let class_attr = last_block.attr("class").await.ok().flatten().unwrap_or_default();

        return match class_attr
            .split_whitespace()
            .find_map(|c| c.strip_prefix("theory-viewer__block_type_"))
        {
            Some("action-button") => {
                match last_block.find(Locator::Css(CONTENT_EXPANDER_SELECTOR)).await {
                    Ok(button) if button.click().await.is_ok() => {
                        Step::Clicked(CONTENT_EXPANDER_SELECTOR.to_string())
                    }
                    _ => Step::Idle,
                }
            }
            Some(QUIZ_SELECT_BLOCK_KIND) if yolo_mode => try_yolo_quiz(&last_block).await,
            Some(other_kind) => Step::Blocked(other_kind.to_string()),
            None => Step::Idle,
        };
    }

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

    let Ok(options) = block.find_all(Locator::Css(QUIZ_OPTION_LABEL_SELECTOR)).await else {
        return Step::Idle;
    };
    if options.is_empty() {
        return Step::Idle;
    }

    let index = rand::random_range(..options.len());
    if options[index].click().await.is_ok() {
        Step::Clicked(format!("yolo: picked random quiz option {}/{}", index + 1, options.len()))
    } else {
        Step::Idle
    }
}
