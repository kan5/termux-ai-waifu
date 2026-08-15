//! Voice assistant — local voice pipeline (Microphone → VAD → STT → LLM → TTS → audio).

#[cfg(not(target_os = "android"))]
mod audio;
#[cfg(target_os = "android")]
mod aaudio;
mod config;
mod llm;
#[cfg(not(target_os = "android"))]
mod pipeline;
mod resample;
mod stt;
#[cfg(target_os = "android")]
mod stream;
mod text;
mod traits;
mod tts;
mod types;
mod vad;

use std::path::PathBuf;

use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let config_path = parse_args();
    let config = config::Config::load(&config_path)
        .with_context(|| format!("failed to load config {}", config_path.display()))?;

    tracing::info!("configuration loaded: {:?}", config);

    run_live(config).await
}

/// Live pipeline: CPAL on Linux/desktop, native AAudio streaming on Termux/Android.
#[cfg(not(target_os = "android"))]
async fn run_live(config: config::Config) -> Result<()> {
    pipeline::run(config).await
}

/// On Android the default (and only) live path is AAudio streaming.
#[cfg(target_os = "android")]
async fn run_live(config: config::Config) -> Result<()> {
    stream::run_forever(config).await
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("voice_assistant=info,info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Parse `--config <path>`.
fn parse_args() -> PathBuf {
    let mut config_path = PathBuf::from("config.toml");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" {
            if let Some(p) = args.next() {
                config_path = PathBuf::from(p);
            }
        }
    }
    config_path
}
