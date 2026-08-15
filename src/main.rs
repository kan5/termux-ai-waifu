//! Voice assistant — local voice pipeline (Microphone → VAD → STT → LLM → TTS → audio).

#[cfg(not(target_os = "android"))]
mod audio;
mod config;
mod llm;
mod offline;
#[cfg(not(target_os = "android"))]
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

    let (config_path, file_args) = parse_args();
    let config = config::Config::load(&config_path)
        .with_context(|| format!("failed to load config {}", config_path.display()))?;

    tracing::info!("configuration loaded: {:?}", config);

    match file_args {
        // Offline (Termux) file pipeline: in.wav → VAD → STT → LLM → TTS → out.wav.
        Some(file_args) => offline::run(config, file_args).await,
        // Live CPAL pipeline (Linux/desktop only).
        #[cfg(not(target_os = "android"))]
        None => pipeline::run(config).await,
        // On Android there is no live pipeline (CPAL can't drive the audio
        // devices), so the file mode is mandatory.
        #[cfg(target_os = "android")]
        None => {
            eprintln!("on Termux/Android the file mode is required: --file-input <wav> --file-output <wav>");
            std::process::exit(2);
        }
    }
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("voice_assistant=info,info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Parse `--config <path>`, `--file-input <wav>`, `--file-output <wav>`.
fn parse_args() -> (PathBuf, Option<offline::FileArgs>) {
    let mut config_path = PathBuf::from("config.toml");
    let mut file_input: Option<PathBuf> = None;
    let mut file_output: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                if let Some(p) = args.next() {
                    config_path = PathBuf::from(p);
                }
            }
            "--file-input" => {
                if let Some(p) = args.next() {
                    file_input = Some(PathBuf::from(p));
                }
            }
            "--file-output" => {
                if let Some(p) = args.next() {
                    file_output = Some(PathBuf::from(p));
                }
            }
            _ => {}
        }
    }

    let file_args = match (file_input, file_output) {
        (Some(input), Some(output)) => Some(offline::FileArgs { input, output }),
        // Only one of the two given — treat as an error at use site.
        (Some(_input), None) => {
            tracing::error!("--file-input given without --file-output; ignoring offline mode");
            eprintln!("--file-input requires --file-output");
            std::process::exit(2);
        }
        (None, Some(_output)) => {
            tracing::error!("--file-output given without --file-input; ignoring offline mode");
            eprintln!("--file-output requires --file-input");
            std::process::exit(2);
        }
        (None, None) => None,
    };

    (config_path, file_args)
}
