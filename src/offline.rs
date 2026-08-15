//! Offline, file-based pipeline for Termux/Android.
//!
//! On Android the CPAL backend cannot drive the microphone/speakers directly,
//! so we route audio through files produced by Termux:API:
//!
//!   record:  termux-microphone-record -f in.m4a          (compressed)
//!            ffmpeg -i in.m4a -ar 16000 -ac 1 in.wav      (→ mono 16k)
//!   play:    termux-media-player play out.wav
//!
//! This module reads `in.wav`, runs VAD → STT → LLM → TTS, and writes the
//! synthesized reply to `out.wav` (mono 16 kHz s16le). It reuses the same
//! components as the live pipeline but over a whole file instead of a stream.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::Config;
use crate::text::filter_think;
use crate::traits::{Llm, SpeechToText, TextToSpeech, Vad, VadState};
use crate::types::AudioChunk;

/// Args for the `--file-input` / `--file-output` offline mode.
pub struct FileArgs {
    pub input: PathBuf,
    pub output: PathBuf,
}

/// Run the whole offline pipeline over a single audio file.
pub async fn run(config: Config, args: FileArgs) -> Result<()> {
    // 1. Read + normalize the input to mono/16k/f32.
    let samples = read_wav_mono_16k(&args.input)?;
    if samples.is_empty() {
        anyhow::bail!("input file {} contains no audio", args.input.display());
    }
    tracing::info!(
        input = %args.input.display(),
        samples = samples.len(),
        seconds = samples.len() as f32 / 16000.0,
        "read input audio"
    );

    // 2. VAD → STT over the whole file.
    let mut vad = crate::vad::SileroVad::new(&config.vad)?;
    let mut stt = crate::stt::GigaamStt::new(&config.stt)?;
    let transcript = transcribe_file(&mut vad, &mut stt, &samples, &config)?;
    let transcript = transcript.trim().to_string();
    if transcript.is_empty() {
        anyhow::bail!("no speech detected in {}", args.input.display());
    }
    tracing::info!(transcript = %transcript, "final transcript");

    // 3. LLM.
    let mut llm = crate::llm::QwenLlm::new(&config.llm)?;
    let mut raw = String::new();
    let mut in_think = false;
    let mut answer = String::new();
    llm.generate(&transcript, &mut |piece| {
        raw.push_str(piece);
        let cleaned = filter_think(piece, &mut in_think);
        answer.push_str(&cleaned);
        true
    })?;
    tracing::info!(llm_output = %raw.trim(), "llm generated");
    if answer.trim().is_empty() {
        anyhow::bail!("LLM produced no answer");
    }

    // 4. TTS → collect PCM.
    let tts = crate::tts::RobynTts::new(&config.tts)?;
    let text = crate::text::normalize_digits(answer.trim());
    tracing::info!(tts_chunk = %text, "synthesizing");
    let audio = tts.synthesize(&text).await?;
    tracing::info!(
        samples = audio.len(),
        seconds = audio.len() as f32 / 16000.0,
        "synthesized reply"
    );

    // 5. Write output WAV.
    write_wav_s16(&args.output, audio.as_slice())?;
    tracing::info!(output = %args.output.display(), "wrote output");
    Ok(())
}

/// Run the VAD state machine over the whole file, accumulating speech and
/// emitting exactly one final transcription at the end.
fn transcribe_file(
    vad: &mut crate::vad::SileroVad,
    stt: &mut crate::stt::GigaamStt,
    samples: &[f32],
    config: &Config,
) -> Result<String> {
    let chunk_samples = config.audio.chunk_samples;
    let mut speech: Vec<f32> = Vec::new();
    let mut prev_chunk: Vec<f32> = Vec::new();
    let mut result = String::new();

    for chunk in samples.chunks(chunk_samples) {
        let chunk = AudioChunk::from_slice(chunk);
        let state = vad.process(&chunk)?;
        match state {
            VadState::SpeechStart => {
                speech.clear();
                speech.extend_from_slice(&prev_chunk); // pre-roll (Russian "р")
                speech.extend_from_slice(chunk.as_slice());
            }
            VadState::Speech => {
                speech.extend_from_slice(chunk.as_slice());
            }
            VadState::SpeechEnd => {
                result = stt.finalize(&speech)?;
                stt.reset();
                speech.clear();
            }
            VadState::Silence => {}
        }
        prev_chunk = chunk.as_slice().to_vec();
    }

    // Trailing speech that never hit SpeechEnd (file ends mid-utterance).
    if !speech.is_empty() {
        result = stt.finalize(&speech)?;
    }
    Ok(result)
}

/// Read a WAV and return it as mono / 16 kHz / f32, converting as needed.
fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        anyhow::bail!("WAV {} has invalid spec {:?}", path.display(), spec);
    }

    // Read samples as f32 regardless of source bit depth.
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let s: Vec<f32> = reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .context("failed to read float samples")?;
            s
        }
        hound::SampleFormat::Int => {
            // Convert int samples to f32 in [-1, 1].
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            let s: Vec<f32> = reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("failed to read int samples")?
                .into_iter()
                .map(|v| v as f32 / max)
                .collect();
            s
        }
    };

    // Downmix to mono (average channels).
    let channels = spec.channels as usize;
    let mut mono: Vec<f32> = Vec::with_capacity(raw.len() / channels);
    for frame in raw.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        mono.push(sum / channels as f32);
    }

    // Resample to 16 kHz if needed.
    if spec.sample_rate != 16000 {
        let mut resampler = crate::resample::LinearResampler::new(spec.sample_rate, 16000);
        mono = resampler.push(&mono);
    }
    Ok(mono)
}

/// Write mono 16 kHz audio as a 16-bit PCM WAV file.
fn write_wav_s16(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV {}", path.display()))?;
    for &s in samples {
        // f32 in [-1, 1] → i16, with clipping guard.
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * 32767.0) as i16;
        writer
            .write_sample(v)
            .context("failed to write WAV sample")?;
    }
    writer
        .finalize()
        .context("failed to finalize WAV")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_wav_s16() {
        let dir = std::env::temp_dir();
        let path = dir.join("va_offline_roundtrip.wav");
        let samples: Vec<f32> = (0..16000).map(|i| ((i as f32 / 16000.0) * 0.5)).collect();
        write_wav_s16(&path, &samples).unwrap();
        let back = read_wav_mono_16k(&path).unwrap();
        // 16k in → 16k out, count ~ preserved (linear resampler tolerance).
        let diff = (back.len() as i64 - samples.len() as i64).abs();
        assert!(diff < 100, "expected ~{} samples, got {}", samples.len(), back.len());
        std::fs::remove_file(&path).ok();
    }
}
