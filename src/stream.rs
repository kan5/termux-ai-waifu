//! Streaming (native AAudio) pipeline for Termux/Android.
//!
//! Captures the mic as a continuous f32 stream (AAudio), feeds VAD chunk by
//! chunk (30 ms), and auto-stops on silence: when VAD reports `SpeechEnd`, the
//! accumulated utterance is finalized through STT → LLM → TTS and played back
//! via AAudio. No temp files, no termux-microphone-record — a true streaming
//! loop like the desktop pipeline.

#![cfg(target_os = "android")]

use anyhow::{Context, Result};
use std::time::Duration;

use crate::config::Config;
use crate::text::filter_think;
use crate::traits::{Llm, SpeechToText, Vad, VadState};
use crate::types::AudioChunk;

/// Run the streaming listen → reply loop forever (until Ctrl-C / signal).
pub async fn run_forever(config: Config) -> Result<()> {
    let chunk_samples = config.audio.chunk_samples; // 480 = 30 ms @ 16k

    loop {
        match run_one_utterance(&config, chunk_samples).await {
            Ok(Some(text)) => {
                tracing::info!(answer = %text, "replied");
            }
            Ok(None) => {
                tracing::info!("no speech detected");
            }
            Err(e) => {
                // Transient failures (e.g. TTS hiccup) shouldn't kill the loop.
                tracing::warn!("utterance failed: {e:#}");
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Listen until one utterance completes (VAD SpeechEnd + trailing silence),
/// run it through LLM→TTS, play the reply, and return the reply text.
async fn run_one_utterance(config: &Config, chunk_samples: usize) -> Result<Option<String>> {
    let mut vad = crate::vad::SileroVad::new(&config.vad)?;
    let mut stt = crate::stt::GigaamStt::new(&config.stt)?;

    let mut input = crate::aaudio::open_input(16000).context("open AAudio input")?;
    crate::aaudio::start(input)?;

    let mut speech: Vec<f32> = Vec::new();
    let mut prev_chunk: Vec<f32> = Vec::new();
    let mut buffer = vec![0f32; chunk_samples];
    let mut finalized: Option<String> = None;

    tracing::info!("listening (stream)...");

    // Listen until we finalize an utterance. AAudio read blocks until frames
    // arrive or timeout; a 100 ms timeout keeps the loop responsive.
    while finalized.is_none() {
        let n = crate::aaudio::read(input, &mut buffer, 100);
        if n < 0 {
            // Timeout (negative) is normal; other errors are fatal.
            if n == -4 {
                // AAUDIO_ERROR_TIMEOUT — no data this round, keep waiting.
                continue;
            }
            anyhow::bail!("AAudio read failed rc={n}");
        }
        if n == 0 {
            continue;
        }
        let chunk = AudioChunk::from_slice(&buffer[..n as usize]);
        let state = vad.process(&chunk)?;
        match state {
            VadState::SpeechStart => {
                speech.clear();
                speech.extend_from_slice(&prev_chunk); // pre-roll
                speech.extend_from_slice(&buffer[..n as usize]);
            }
            VadState::Speech => {
                speech.extend_from_slice(&buffer[..n as usize]);
            }
            VadState::SpeechEnd => {
                let text = stt.finalize(&speech)?;
                stt.reset();
                speech.clear();
                finalized = Some(text);
            }
            VadState::Silence => {}
        }
        prev_chunk = buffer[..n as usize].to_vec();
    }

    crate::aaudio::close(input);

    let text = finalized.unwrap().trim().to_string();
    if text.is_empty() {
        return Ok(None);
    }
    tracing::info!(transcript = %text, "final transcript");

    // LLM.
    let mut llm = crate::llm::QwenLlm::new(&config.llm)?;
    let mut raw = String::new();
    let mut in_think = false;
    let mut answer = String::new();
    llm.generate(&text, &mut |piece| {
        raw.push_str(piece);
        answer.push_str(&filter_think(piece, &mut in_think));
        true
    })?;
    tracing::info!(llm_output = %raw.trim(), "llm generated");
    let answer = answer.trim();
    if answer.is_empty() {
        return Ok(None);
    }

    // TTS.
    let tts = crate::tts::RobynTts::new(&config.tts)?;
    let text = crate::text::normalize_digits(answer);
    tracing::info!(tts_chunk = %text, "synthesizing");
    let audio = tts.synthesize(&text).await?;
    tracing::info!(samples = audio.len(), "synthesized reply");

    // Play via AAudio (blocking, on a blocking task).
    let play_audio = audio.clone();
    tokio::task::spawn_blocking(move || {
        crate::aaudio::play(&play_audio.as_slice(), 16000)
    })
    .await
    .context("AAudio play task panicked")??;

    Ok(Some(answer.to_string()))
}
