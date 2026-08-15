//! Voice assistant — local voice pipeline (Microphone → VAD → STT → LLM → TTS → audio).

mod audio;
mod config;
mod llm;
mod pipeline;
mod stt;
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

    pipeline::run(config).await
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("voice_assistant=info,info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Returns the config path from `--config <path>`, or `config.toml`.
fn parse_args() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
        }
    }
    PathBuf::from("config.toml")
}
