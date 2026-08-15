//! Component contracts.
//!
//! Every ML backend (VAD / STT / LLM / TTS) is hidden behind a trait so that
//! a concrete implementation can be swapped out without touching the rest of
//! the application (ТЗ §9). The pipeline is generic over these traits.

use anyhow::Result;
use tokio::sync::mpsc;

use crate::types::{AudioChunk, PlaybackCommand, TranscriptEvent};

/// State transition reported by a VAD for a single audio chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadState {
    /// No speech detected.
    Silence,
    /// Rising edge — speech just began. Used for barge-in and to open a new utterance.
    SpeechStart,
    /// Speech in progress; audio should keep flowing to STT.
    Speech,
    /// Falling edge — speech ended; the utterance is complete.
    SpeechEnd,
}

/// Voice activity detection (Silero VAD).
///
/// Implementations are expected to keep their own state machine
/// (min speech / min silence durations, padding) internally.
pub trait Vad: Send {
    /// Feed one audio chunk and return the resulting state.
    fn process(&mut self, chunk: &AudioChunk) -> Result<VadState>;
    /// Reset internal state for a fresh utterance.
    fn reset(&mut self);
}

/// Speech-to-text (GigaAM via transcribe.cpp).
///
/// The chosen model is not a true streaming model, so implementations re-run
/// recognition over accumulated audio to expose partial hypotheses.
pub trait SpeechToText: Send {
    /// Transcribe accumulated audio and return the current best hypothesis.
    fn transcribe(&mut self, audio: &[f32]) -> Result<String>;
    /// Final transcription of a completed utterance.
    fn finalize(&mut self, audio: &[f32]) -> Result<String>;
    /// Reset for the next utterance.
    fn reset(&mut self);
}

/// Streaming LLM (Qwen via llama.cpp).
///
/// `on_token` is called with each text fragment as it is generated; returning
/// `false` requests early termination (used for barge-in).
pub trait Llm: Send {
    fn generate(&mut self, prompt: &str, on_token: &mut dyn FnMut(&str) -> bool) -> Result<()>;
}

/// Text-to-speech. Returns mono 16 kHz f32 audio for the given text.
///
/// Backed by a separate Python (Silero TTS) HTTP service.
pub trait TextToSpeech: Send + Sync {
    async fn synthesize(&self, text: &str) -> Result<AudioChunk>;
}

/// Audio capture (microphone → mono 16 kHz f32 chunks).
///
/// Intended to run on a blocking thread; sends chunks into a Tokio channel.
pub trait AudioCapture: Send {
    fn stream(self: Box<Self>, tx: mpsc::Sender<AudioChunk>) -> Result<()>;
}

/// Audio playback (mono 16 kHz f32 chunks → output device).
///
/// Runs on a blocking thread; drains the channel until it is closed. A
/// [`PlaybackCommand::Flush`] clears buffered audio for barge-in.
pub trait AudioSink: Send {
    fn play(self: Box<Self>, rx: mpsc::Receiver<PlaybackCommand>) -> Result<()>;
}

/// Convenience alias reused by the pipeline controller.
pub type TranscriptSender = mpsc::Sender<TranscriptEvent>;
pub type TranscriptReceiver = mpsc::Receiver<TranscriptEvent>;
