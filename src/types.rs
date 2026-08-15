//! Shared data types flowing between pipeline components.
//!
//! The internal audio format is fixed by ТЗ §4: **mono / 16 kHz / f32**.
//! Conversion to/from device formats happens only at the edges (audio module).

use std::sync::Arc;

/// A chunk of mono 16 kHz f32 audio samples.
///
/// `Arc<[f32]>` makes chunks cheap to clone across broadcast channels
/// (e.g. fanning the mic stream out to VAD and barge-in detection).
#[derive(Clone, Debug, Default)]
pub struct AudioChunk(Arc<[f32]>);

impl AudioChunk {
    pub fn new(samples: Vec<f32>) -> Self {
        Self(samples.into())
    }

    pub fn from_slice(samples: &[f32]) -> Self {
        Self(samples.into())
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<f32>> for AudioChunk {
    fn from(v: Vec<f32>) -> Self {
        Self(v.into())
    }
}

/// Transcript events emitted by STT toward the pipeline controller.
#[derive(Clone, Debug)]
pub enum TranscriptEvent {
    /// Intermediate hypothesis while the user is still speaking.
    Partial(String),
    /// Final transcript of a completed utterance — triggers the LLM.
    Final(String),
}

/// Control messages sent to the audio sink.
#[derive(Clone, Debug)]
pub enum PlaybackCommand {
    /// Play this chunk of audio.
    Play(AudioChunk),
    /// Immediately drop buffered audio and resume silence (barge-in stop).
    Flush,
}
