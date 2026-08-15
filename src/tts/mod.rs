//! Text-to-speech via a separate Python (Silero TTS) HTTP service.
//!
//! The Rust side POSTs `{"text", "speaker"}` to the service and receives raw
//! little-endian f32 PCM (mono, 16 kHz) as the response body, which is wrapped
//! directly into an [`AudioChunk`].

use anyhow::{Context, Result};
use reqwest::Client;

use crate::config::TtsConfig;
use crate::traits::TextToSpeech;
use crate::types::AudioChunk;

pub struct RobynTts {
    client: Client,
    url: String,
    speaker: String,
}

impl RobynTts {
    pub fn new(cfg: &TtsConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(cfg.timeout_secs))
            .build()
            .context("failed to build TTS HTTP client")?;
        Ok(Self {
            client,
            url: cfg.url.clone(),
            speaker: cfg.speaker.clone(),
        })
    }
}

impl TextToSpeech for RobynTts {
    async fn synthesize(&self, text: &str) -> Result<AudioChunk> {
        let body = serde_json::json!({
            "text": text,
            "speaker": self.speaker,
        });
        let resp = self
            .client
            .post(format!("{}/tts", self.url))
            .json(&body)
            .send()
            .await
            .context("TTS request failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("TTS service returned {}", resp.status());
        }

        let bytes = resp.bytes().await.context("TTS response read failed")?;
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(AudioChunk::new(samples))
    }
}
