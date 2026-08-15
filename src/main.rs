//! Voice assistant — local voice pipeline (Microphone → VAD → STT → LLM → TTS → audio).

#[cfg(not(target_os = "android"))]
mod audio;
#[cfg(target_os = "android")]
mod aaudio;
mod config;
mod llm;
mod offline;
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

    let (config_path, mode) = parse_args();
    let config = config::Config::load(&config_path)
        .with_context(|| format!("failed to load config {}", config_path.display()))?;

    tracing::info!("configuration loaded: {:?}", config);

    match mode {
        // Streaming native AAudio loop (Termux/Android): mic → VAD → STT → LLM → TTS → speaker.
        #[cfg(target_os = "android")]
        Mode::Stream => stream::run_forever(config).await,
        // Offline (Termux) file pipeline: in.wav → VAD → STT → LLM → TTS → out.wav.
        Mode::File(file_args) => offline::run(config, file_args).await,
        // Live CPAL pipeline (Linux/desktop only).
        #[cfg(not(target_os = "android"))]
        Mode::Live => pipeline::run(config).await,
        // On Android there is no live CPAL pipeline, so the default is streaming.
        #[cfg(target_os = "android")]
        Mode::Live => stream::run_forever(config).await,
    }
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("voice_assistant=info,info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Runtime mode selected on the CLI.
enum Mode {
    /// Offline file pipeline (`--file-input` / `--file-output`).
    File(offline::FileArgs),
    /// Live pipeline. On desktop this is CPAL; on Android it is AAudio streaming.
    Live,
    /// Streaming native AAudio loop (Termux/Android). `--stream`.
    #[cfg(target_os = "android")]
    Stream,
}

/// Parse `--config <path>`, `--file-input <wav>`, `--file-output <wav>`, `--stream`.
fn parse_args() -> (PathBuf, Mode) {
    let mut config_path = PathBuf::from("config.toml");
    let mut file_input: Option<PathBuf> = None;
    let mut file_output: Option<PathBuf> = None;
    #[cfg(target_os = "android")]
    let mut stream = false;

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
            #[cfg(target_os = "android")]
            "--stream" => stream = true,
            _ => {}
        }
    }

    #[cfg(target_os = "android")]
    if stream {
        // --stream overrides file args if both given.
        return (config_path, Mode::Stream);
    }

    match (file_input, file_output) {
        (Some(input), Some(output)) => (config_path, Mode::File(offline::FileArgs { input, output })),
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
        (None, None) => (config_path, Mode::Live),
    }
}
