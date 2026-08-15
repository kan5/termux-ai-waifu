//! Pipeline orchestration.
//!
//! Component graph (all concurrent, wired over Tokio channels):
//!
//! ```text
//! capture (blocking) ──AudioChunk──▶ VAD+STT driver (blocking)
//!                                        │ ControlEvent (Partial/Final/SpeechStarted)
//!                                        ▼
//!                                   controller (async)
//!                                        │ Final(text) ──▶ LLM (blocking) ─text─▶ TTS (async) ─audio─▶ playback
//!                                        │                                                              ▲
//!                                        └── SpeechStarted → bump generation → Flush ───────────────────┘
//! ```
//!
//! Barge-in: when the VAD hears a new `SpeechStart` while a response is in
//! flight, the controller bumps a shared generation counter and flushes the
//! playback queue. The LLM callback and the TTS loop both check the generation
//! and abort when it changes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::audio::{CpalCapture, CpalSink};
use crate::config::{Config, LlmConfig, SttConfig};
use crate::llm::QwenLlm;
use crate::stt::GigaamStt;
use crate::traits::{AudioCapture, AudioSink, Llm, SpeechToText, TextToSpeech, Vad, VadState};
use crate::tts::RobynTts;
use crate::types::{AudioChunk, PlaybackCommand, TranscriptEvent};
use crate::vad::SileroVad;

/// Events flowing from the VAD/STT driver to the controller.
pub enum ControlEvent {
    Transcript(TranscriptEvent),
    /// A new utterance started — used for barge-in.
    SpeechStarted,
}

pub async fn run(config: Config) -> Result<()> {
    let capture = Box::new(CpalCapture::new(config.audio.clone()));
    let sink = Box::new(CpalSink::new(config.audio.clone()));

    let vad = SileroVad::new(&config.vad)?;
    let stt = GigaamStt::new(&config.stt)?;
    let llm = Arc::new(Mutex::new(QwenLlm::new(&config.llm)?));
    let tts = Arc::new(RobynTts::new(&config.tts)?);

    let (tx_audio, rx_audio) = mpsc::channel::<AudioChunk>(256);
    let (tx_control, rx_control) = mpsc::channel::<ControlEvent>(64);
    let (tx_playback, rx_playback) = mpsc::channel::<PlaybackCommand>(256);

    let generation = Arc::new(AtomicU64::new(0));

    // Persistent playback sink.
    tokio::task::spawn_blocking(move || {
        if let Err(e) = sink.play(rx_playback) {
            tracing::error!("playback error: {e:#}");
        }
    });

    // Microphone capture.
    tokio::task::spawn_blocking(move || {
        if let Err(e) = capture.stream(tx_audio) {
            tracing::error!("capture error: {e:#}");
        }
    });

    // VAD + STT driver.
    let stt_cfg = config.stt.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = vad_stt_driver(vad, stt, rx_audio, tx_control, stt_cfg) {
            tracing::error!("VAD/STT driver error: {e:#}");
        }
    });

    // Controller.
    let llm_cfg = config.llm.clone();
    tokio::spawn(controller(
        rx_control,
        llm,
        tts,
        tx_playback,
        generation,
        llm_cfg,
    ));

    tracing::info!("voice assistant running — press Ctrl-C to stop");
    tokio::signal::ctrl_c()
        .await
        .context("failed to listen for Ctrl-C")?;
    tracing::info!("shutting down");
    // Blocking audio/ML tasks are not cooperatively cancellable, so exit
    // directly; the OS releases the audio devices.
    std::process::exit(0);
}

/// Feeds audio into the VAD, accumulates speech, drives STT partials, and emits
/// control events. Runs on a blocking thread (CPU-bound VAD + STT).
fn vad_stt_driver(
    mut vad: SileroVad,
    mut stt: GigaamStt,
    mut rx_audio: mpsc::Receiver<AudioChunk>,
    tx_control: mpsc::Sender<ControlEvent>,
    stt_cfg: SttConfig,
) -> Result<()> {
    let partial_interval = Duration::from_millis(stt_cfg.partial_interval_ms);
    let mut speech_buf: Vec<f32> = Vec::new();
    let mut prev_chunk: Vec<f32> = Vec::new();
    let mut last_partial = Instant::now();

    while let Some(chunk) = rx_audio.blocking_recv() {
        let state = vad.process(&chunk)?;
        match state {
            VadState::SpeechStart => {
                speech_buf.clear();
                // Pre-roll: include one chunk before the detected onset so a
                // low-energy word-initial consonant (e.g. the Russian "р")
                // isn't clipped by the VAD's speech-start detection.
                speech_buf.extend_from_slice(&prev_chunk);
                speech_buf.extend_from_slice(chunk.as_slice());
                last_partial = Instant::now();
                if tx_control.blocking_send(ControlEvent::SpeechStarted).is_err() {
                    break;
                }
                if let Ok(t) = stt.transcribe(&speech_buf) {
                    let _ = tx_control
                        .blocking_send(ControlEvent::Transcript(TranscriptEvent::Partial(t)));
                }
            }
            VadState::Speech => {
                speech_buf.extend_from_slice(chunk.as_slice());
                if last_partial.elapsed() >= partial_interval {
                    last_partial = Instant::now();
                    if let Ok(t) = stt.transcribe(&speech_buf) {
                        let _ = tx_control
                            .blocking_send(ControlEvent::Transcript(TranscriptEvent::Partial(t)));
                    }
                }
            }
            VadState::SpeechEnd => {
                let final_text = stt.finalize(&speech_buf)?;
                stt.reset();
                speech_buf.clear();
                if tx_control
                    .blocking_send(ControlEvent::Transcript(TranscriptEvent::Final(final_text)))
                    .is_err()
                {
                    break;
                }
            }
            VadState::Silence => {}
        }
        prev_chunk = chunk.as_slice().to_vec();
    }
    Ok(())
}

/// Central controller: reacts to transcripts and barge-in events.
async fn controller(
    mut rx_control: mpsc::Receiver<ControlEvent>,
    llm: Arc<Mutex<QwenLlm>>,
    tts: Arc<RobynTts>,
    tx_playback: mpsc::Sender<PlaybackCommand>,
    generation: Arc<AtomicU64>,
    llm_cfg: LlmConfig,
) -> Result<()> {
    while let Some(ev) = rx_control.recv().await {
        match ev {
            ControlEvent::SpeechStarted => {
                // Barge-in: cancel the in-flight response and silence playback.
                generation.fetch_add(1, Ordering::SeqCst);
                let _ = tx_playback.try_send(PlaybackCommand::Flush);
            }
            ControlEvent::Transcript(TranscriptEvent::Partial(text)) => {
                tracing::debug!(partial = %text, "partial transcript");
            }
            ControlEvent::Transcript(TranscriptEvent::Final(text)) => {
                let text = text.trim().to_string();
                if text.is_empty() {
                    tracing::debug!("empty final transcript, skipping");
                    continue;
                }
                tracing::info!(transcript = %text, "final transcript");
                spawn_response(
                    Arc::clone(&llm),
                    Arc::clone(&tts),
                    tx_playback.clone(),
                    Arc::clone(&generation),
                    text,
                    llm_cfg.clone(),
                );
            }
        }
    }
    Ok(())
}

/// Run LLM → TTS → playback for one response, with barge-in cancellation.
fn spawn_response(
    llm: Arc<Mutex<QwenLlm>>,
    tts: Arc<RobynTts>,
    tx_playback: mpsc::Sender<PlaybackCommand>,
    generation: Arc<AtomicU64>,
    text: String,
    llm_cfg: LlmConfig,
) {
    let my_generation = generation.load(Ordering::SeqCst);
    tokio::spawn(async move {
        let (tx_text, mut rx_text) = mpsc::channel::<String>(32);

        // LLM generation (blocking) → text chunks.
        let llm_task = {
            let generation = Arc::clone(&generation);
            tokio::task::spawn_blocking(move || {
                let mut llm = llm.lock().unwrap();
                let target = llm_cfg.chunk_char_target;
                let mut buf = String::new();
                let mut raw_out = String::new();
                let mut in_think = false;
                let result = llm.generate(&text, &mut |piece| {
                    if generation.load(Ordering::SeqCst) != my_generation {
                        return false; // barge-in
                    }
                    raw_out.push_str(piece);
                    buf.push_str(piece);
                    if buf.chars().count() >= target || ends_sentence(&buf) {
                        let chunk = std::mem::take(&mut buf);
                        let cleaned = filter_think(&chunk, &mut in_think);
                        if !cleaned.trim().is_empty()
                            && tx_text.blocking_send(cleaned).is_err()
                        {
                            return false;
                        }
                    }
                    true
                });
                if !buf.is_empty() {
                    let cleaned = filter_think(&buf, &mut in_think);
                    if !cleaned.trim().is_empty() {
                        let _ = tx_text.blocking_send(cleaned);
                    }
                }
                tracing::info!(llm_output = %raw_out.trim(), "llm generated");
                // Dropping tx_text closes the channel and ends the TTS loop.
                drop(tx_text);
                result
            })
        };

        // TTS (async) → playback.
        while let Some(chunk) = rx_text.recv().await {
            if generation.load(Ordering::SeqCst) != my_generation {
                break; // barge-in
            }
            let chunk = crate::text::normalize_digits(chunk.trim());
            if chunk.is_empty() {
                continue; // skip whitespace-only fragments (e.g. leading newline)
            }
            tracing::debug!(tts_chunk = %chunk, "synthesizing");
            match tts.synthesize(&chunk).await {
                Ok(audio) => {
                    if tx_playback.send(PlaybackCommand::Play(audio)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("TTS failed: {e:#}");
                    break;
                }
            }
        }

        if let Err(e) = llm_task.await {
            tracing::error!("LLM task error: {e:#}");
        }
    });
}

/// True when the accumulated text ends on a sentence boundary.
fn ends_sentence(s: &str) -> bool {
    let t = s.trim_end();
    t.ends_with('.') || t.ends_with('!') || t.ends_with('?') || t.ends_with('…') || t.ends_with('\n')
}

/// Strip Qwen3's `<think>…</think>` reasoning blocks from a text fragment.
/// `in_think` persists across calls so blocks spanning chunk boundaries are
/// handled correctly.
pub(crate) fn filter_think(chunk: &str, in_think: &mut bool) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let mut out = String::new();
    let mut rest = chunk;
    loop {
        if *in_think {
            match rest.find(CLOSE) {
                Some(idx) => {
                    *in_think = false;
                    rest = &rest[idx + CLOSE.len()..];
                }
                None => break, // still inside a think block — drop the rest
            }
        } else {
            match rest.find(OPEN) {
                Some(idx) => {
                    out.push_str(&rest[..idx]);
                    *in_think = true;
                    rest = &rest[idx + OPEN.len()..];
                }
                None => {
                    out.push_str(rest);
                    break;
                }
            }
        }
    }
    out
}
