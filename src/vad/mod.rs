//! Voice activity detection via Silero VAD (ONNX Runtime).
//!
//! The v5 streaming model takes `input` (context + frame audio), `state`
//! (LSTM hidden state) and `sr` (sample rate) and returns a speech
//! probability plus the next `state`. Frame size is 512 samples (32 ms) at
//! 16 kHz; context is 64 samples. The I/O contract follows the official
//! Silero VAD Rust example.
//!
//! Short utterances (< min_speech_ms) are not filtered here — the pipeline
//! drops utterances whose final transcript is empty.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayD};
use ort::session::Session;
use ort::value::Value;

use crate::config::VadConfig;
use crate::traits::{Vad, VadState};
use crate::types::AudioChunk;

const SAMPLE_RATE: i64 = 16000;
/// Samples per inference frame at 16 kHz (32 ms).
const FRAME_SIZE: usize = 512;
/// Context samples prepended to each frame at 16 kHz.
const CONTEXT_SIZE: usize = 64;
/// Hidden state shape `[2, 1, 128]`.
const STATE_SHAPE: [usize; 3] = [2, 1, 128];

pub struct SileroVad {
    session: Session,
    sr: Array1<i64>,
    state: ArrayD<f32>,
    context: Array1<f32>,
    /// Incoming samples buffered until a full 512-sample frame is available.
    frame_buf: Vec<f32>,
    threshold: f32,
    min_silence_samples: usize,
    // State machine.
    in_speech: bool,
    speech_start_sample: usize,
    silence_start_sample: usize,
    current_sample: usize,
}

impl SileroVad {
    pub fn new(cfg: &VadConfig) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create ONNX session builder")?
            .commit_from_file(&cfg.model_path)
            .with_context(|| format!("failed to load VAD model {}", cfg.model_path.display()))?;
        let sr = Array1::from_vec(vec![SAMPLE_RATE]);
        let state = ArrayD::<f32>::zeros(STATE_SHAPE.as_slice());
        let context = Array1::<f32>::zeros(CONTEXT_SIZE);
        let sr_per_ms = SAMPLE_RATE as usize / 1000;
        Ok(Self {
            session,
            sr,
            state,
            context,
            frame_buf: Vec::with_capacity(FRAME_SIZE * 2),
            threshold: cfg.threshold,
            min_silence_samples: sr_per_ms * cfg.min_silence_ms as usize,
            in_speech: false,
            speech_start_sample: 0,
            silence_start_sample: 0,
            current_sample: 0,
        })
    }

    /// Run the model on one 512-sample frame, returning the speech probability.
    fn calc_level(&mut self, frame: &[f32]) -> Result<f32> {
        let mut input = Vec::with_capacity(CONTEXT_SIZE + frame.len());
        input.extend_from_slice(self.context.as_slice().unwrap());
        input.extend_from_slice(frame);
        let input_arr = Array2::from_shape_vec([1, input.len()], input)
            .context("failed to build VAD input tensor")?;
        let frame_value = Value::from_array(input_arr).context("failed to wrap VAD input")?;
        let state_value =
            Value::from_array(std::mem::take(&mut self.state)).context("failed to wrap VAD state")?;
        let sr_value = Value::from_array(self.sr.clone()).context("failed to wrap VAD sr")?;

        let res = self
            .session
            .run([
                (&frame_value).into(),
                (&state_value).into(),
                (&sr_value).into(),
            ])
            .context("VAD inference failed")?;

        let (shape, state_data) = res["stateN"]
            .try_extract_tensor::<f32>()
            .context("VAD output missing 'stateN'")?;
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        self.state = ArrayD::from_shape_vec(shape_usize.as_slice(), state_data.to_vec())
            .context("failed to rebuild VAD state")?;

        if frame.len() >= CONTEXT_SIZE {
            self.context = Array1::from_vec(frame[frame.len() - CONTEXT_SIZE..].to_vec());
        }

        let prob = res["output"]
            .try_extract_tensor::<f32>()
            .context("VAD output missing 'output'")?
            .1[0];
        Ok(prob)
    }

    /// Advance the state machine by one frame and return the resulting state.
    fn step(&mut self, prob: f32) -> VadState {
        self.current_sample += FRAME_SIZE;
        if prob > self.threshold {
            self.silence_start_sample = 0;
            if !self.in_speech {
                self.in_speech = true;
                self.speech_start_sample = self.current_sample - FRAME_SIZE;
                VadState::SpeechStart
            } else {
                VadState::Speech
            }
        } else if self.in_speech {
            if self.silence_start_sample == 0 {
                self.silence_start_sample = self.current_sample;
            }
            if self.current_sample - self.silence_start_sample >= self.min_silence_samples {
                self.in_speech = false;
                self.silence_start_sample = 0;
                VadState::SpeechEnd
            } else {
                VadState::Speech
            }
        } else {
            VadState::Silence
        }
    }
}

impl Vad for SileroVad {
    fn process(&mut self, chunk: &AudioChunk) -> Result<VadState> {
        self.frame_buf.extend_from_slice(chunk.as_slice());
        let mut result = VadState::Silence;
        while self.frame_buf.len() >= FRAME_SIZE {
            let frame: Vec<f32> = self.frame_buf.drain(..FRAME_SIZE).collect();
            let prob = self.calc_level(&frame)?;
            result = merge(result, self.step(prob));
        }
        Ok(result)
    }

    fn reset(&mut self) {
        self.frame_buf.clear();
        self.state = ArrayD::<f32>::zeros(STATE_SHAPE.as_slice());
        self.context = Array1::<f32>::zeros(CONTEXT_SIZE);
        self.in_speech = false;
        self.speech_start_sample = 0;
        self.silence_start_sample = 0;
        self.current_sample = 0;
    }
}

/// Collapse several per-frame states into the single most significant one.
/// Priority: SpeechEnd > SpeechStart > Speech > Silence.
fn merge(a: VadState, b: VadState) -> VadState {
    use VadState::*;
    match (a, b) {
        (SpeechEnd, _) | (_, SpeechEnd) => SpeechEnd,
        (SpeechStart, _) | (_, SpeechStart) => SpeechStart,
        (Speech, _) | (_, Speech) => Speech,
        _ => Silence,
    }
}
