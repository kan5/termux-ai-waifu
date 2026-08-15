//! Speech-to-text via transcribe.cpp (GigaAM GGUF).
//!
//! GigaAM is not a true streaming model, so partial transcripts are produced by
//! re-running recognition over the accumulated speech audio (ТЗ §2). The
//! `SpeechToText` trait keeps this module swappable for a real streaming model
//! later.

use anyhow::{Context, Result};
use transcribe_cpp::{Model, RunOptions, Session};

use crate::config::SttConfig;
use crate::traits::SpeechToText;

pub struct GigaamStt {
    /// Kept alive for as long as `session` references the native model.
    _model: Model,
    session: Session,
    language: Option<String>,
}

impl GigaamStt {
    pub fn new(cfg: &SttConfig) -> Result<Self> {
        let model = Model::load(&cfg.model_path)
            .with_context(|| format!("failed to load STT model {}", cfg.model_path.display()))?;
        let session = model.session().context("failed to create STT session")?;
        Ok(Self {
            _model: model,
            session,
            language: (!cfg.language.is_empty()).then(|| cfg.language.clone()),
        })
    }

    fn run_options(&self) -> RunOptions {
        RunOptions {
            language: self.language.clone(),
            ..Default::default()
        }
    }
}

impl SpeechToText for GigaamStt {
    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let opts = self.run_options();
        let result = self.session.run(audio, &opts).context("STT run failed")?;
        Ok(result.text)
    }

    fn finalize(&mut self, audio: &[f32]) -> Result<String> {
        self.transcribe(audio)
    }

    fn reset(&mut self) {
        // GigaAM runs are stateless: every `run` transcribes the buffer it is
        // given, so there is no per-utterance state to clear. A streaming model
        // (swap target) would reset its stream here.
    }
}
