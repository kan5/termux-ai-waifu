//! Application configuration.
//!
//! Loaded from a TOML file (default `config.toml`). Every ML backend gets its
//! own section so that components stay swappable.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub vad: VadConfig,
    #[serde(default)]
    pub stt: SttConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub tts: TtsConfig,
}

/// Internal audio format is always mono / 16 kHz / f32 (see ТЗ §4).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    /// Explicit device name; `None` means "default input/output device".
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Size of one audio chunk in samples at `sample_rate`.
    /// 16000 * 0.030 = 480 samples (30 ms) — typical for VAD.
    pub chunk_samples: usize,
    /// Override the capture device's sample rate (None = auto-detect). Useful
    /// when the OS reports a rate that doesn't match the real device.
    pub input_sample_rate: Option<u32>,
    /// Override the playback device's sample rate (None = auto-detect).
    pub output_sample_rate: Option<u32>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_device: None,
            output_device: None,
            sample_rate: 16000,
            channels: 1,
            chunk_samples: 480,
            input_sample_rate: None,
            output_sample_rate: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VadConfig {
    /// Path to `silero_vad.onnx`.
    pub model_path: PathBuf,
    /// Speech probability threshold in [0, 1].
    pub threshold: f32,
    /// Min speech duration (ms) before triggering "speech start".
    pub min_speech_ms: u32,
    /// Silence duration (ms) before triggering "speech end".
    pub min_silence_ms: u32,
    /// Padding (ms) kept around detected speech before sending to STT.
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/silero_vad.onnx"),
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 500,
            speech_pad_ms: 300,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SttConfig {
    /// Path to the GigaAM GGUF model.
    pub model_path: PathBuf,
    /// Language hint passed to transcribe.cpp (e.g. "ru").
    pub language: String,
    /// Run STT on every N ms of accumulated speech audio (partials).
    pub partial_interval_ms: u64,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/gigaam-v3-e2e-ctc-Q8_0.gguf"),
            language: "ru".to_string(),
            partial_interval_ms: 500,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// Path to the Qwen GGUF model.
    pub model_path: PathBuf,
    /// Context size in tokens.
    pub n_ctx: u32,
    /// Max tokens to generate per answer.
    pub max_tokens: u32,
    pub temperature: f32,
    pub system_prompt: String,
    /// Max characters to accumulate into a TTS chunk before flushing.
    pub chunk_char_target: usize,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/Qwen3-0.6B-abliterated-q4_k_m.gguf"),
            n_ctx: 2048,
            max_tokens: 512,
            temperature: 0.7,
            system_prompt: "Ты — голосовой ассистент. Отвечай кратко, на русском, без разметки.".to_string(),
            chunk_char_target: 160,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    /// Base URL of the Python Robyn TTS service.
    pub url: String,
    /// Silero speaker voice id.
    pub speaker: String,
    /// Sample rate produced by the TTS service (must match `audio`).
    pub sample_rate: u32,
    pub timeout_secs: u64,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8090".to_string(),
            speaker: "xenia".to_string(),
            sample_rate: 16000,
            timeout_secs: 120,
        }
    }
}

impl Config {
    /// Load config from `path`. Missing fields fall back to defaults.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        let cfg: Config = toml::from_str(&text)
            .with_context(|| format!("failed to parse config file {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.vad.threshold) {
            anyhow::bail!("vad.threshold must be within [0, 1], got {}", self.vad.threshold);
        }
        Ok(())
    }
}
