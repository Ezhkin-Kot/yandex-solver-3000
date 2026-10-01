//! Minimal client for a local Ollama server, used by `practice::try_solve`
//! to generate practice-task solutions. Sends a single prompt, returns
//! plain text; no conversation state kept between calls.
//!
//! Unlike `geckodriver` (always spawned by us), Ollama may or may not
//! already be running as a service the user manages themselves — so
//! `ensure_running` only starts it (and hands back a `Child` for the
//! caller to clean up) when nothing answers on its port yet, and leaves an
//! already-running server alone.

use std::error::Error;
use std::process::Stdio;
use std::time::{Duration, Instant};

use serde_json::Value as Json;
use tokio::process::{Child, Command};
use tokio::time::sleep;

const OLLAMA_BASE_URL: &str = "http://localhost:11434";
const OLLAMA_READY_TIMEOUT: Duration = Duration::from_secs(15);
const OLLAMA_READY_POLL_INTERVAL: Duration = Duration::from_millis(300);

/// If a server is already reachable, does nothing and returns `Ok(None)` —
/// it isn't ours to manage, so we won't kill it on exit either. Otherwise
/// spawns `ollama serve` ourselves and waits for it to become reachable,
/// returning the child process for the caller to kill when done (see
/// `spawn_geckodriver`/its cleanup in `main.rs` for the same pattern).
pub async fn ensure_running() -> Result<Option<Child>, Box<dyn Error>> {
    let client = reqwest::Client::new();
    if is_reachable(&client).await {
        return Ok(None);
    }

    println!("Starting ollama serve...");
    let child = Command::new("ollama")
        .arg("serve")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("failed to start `ollama serve` ({e}); is Ollama installed and on PATH?"))?;

    let deadline = Instant::now() + OLLAMA_READY_TIMEOUT;
    while !is_reachable(&client).await {
        if Instant::now() >= deadline {
            return Err("`ollama serve` did not become ready in time".into());
        }
        sleep(OLLAMA_READY_POLL_INTERVAL).await;
    }
    Ok(Some(child))
}

async fn is_reachable(client: &reqwest::Client) -> bool {
    client
        .get(format!("{OLLAMA_BASE_URL}/api/version"))
        .send()
        .await
        .is_ok()
}

/// Stops an `ollama serve` process previously returned by `ensure_running`.
///
/// `ollama serve` spawns its own child process per loaded model
/// (`llama-server`, which is what actually holds the multi-GB model in
/// memory) — confirmed live that killing just the parent orphans that
/// child, which keeps running indefinitely. So children are killed by
/// parent PID first, then the `ollama serve` process itself.
pub async fn stop(mut child: Child) {
    if let Some(pid) = child.id() {
        let _ = std::process::Command::new("pkill")
            .args(["-P", &pid.to_string()])
            .status();
    }
    let _ = child.kill().await;
}

/// Sends `prompt` to a local Ollama server at the given sampling
/// `temperature` and returns the model's response text. Low by default —
/// these prompts ask for one specific, narrow piece of code (often with an
/// exact expression already spelled out in a hint), not open-ended writing
/// — confirmed live that the default (higher) sampling temperature made
/// `qwen2.5-coder:3b` "elaborate" with unrequested intermediate variables
/// and formatting even when the hint gave the literal expression to use,
/// failing the check every time. `temperature` is caller-controlled rather
/// than a fixed constant so `practice::try_solve` can raise it when the
/// model turns out to be repeating the exact same (wrong) answer despite
/// different corrective feedback each time — see its `STUCK_TEMPERATURE`.
pub async fn complete(model: &str, prompt: &str, temperature: f32) -> Result<String, Box<dyn Error>> {
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{OLLAMA_BASE_URL}/api/generate"))
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": temperature,
            },
        }))
        .send()
        .await
        .map_err(|e| format!("couldn't reach Ollama at {OLLAMA_BASE_URL} ({e})"))?;

    let status = response.status();
    let body: Json = response.json().await?;

    if !status.is_success() {
        let message = body
            .get("error")
            .and_then(Json::as_str)
            .unwrap_or("unknown error");
        return Err(format!("Ollama error ({status}): {message}").into());
    }

    body.get("response")
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("Ollama response had no text content: {body}").into())
}
