//! Streaming linear-interpolation resampler for mono f32 audio.
//!
//! Converts between device sample rate and the internal 16 kHz. Keeps one
//! sample of overlap across `push` calls for sample-accurate continuity.

use std::collections::VecDeque;

pub struct LinearResampler {
    /// Input samples consumed per output sample.
    step: f64,
    /// Absolute input position (float) of the next output sample.
    phase: f64,
    /// Recent input samples, oldest first.
    buf: VecDeque<f32>,
}

impl LinearResampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        assert!(input_rate > 0 && output_rate > 0, "sample rates must be positive");
        Self {
            step: input_rate as f64 / output_rate as f64,
            phase: 0.0,
            buf: VecDeque::new(),
        }
    }

    /// Feed input samples and return any newly produced output samples.
    pub fn push(&mut self, input: &[f32]) -> Vec<f32> {
        self.buf.extend(input.iter().copied());
        let mut out = Vec::new();
        loop {
            let i = self.phase.floor() as usize;
            // Linear interpolation needs samples at `i` and `i + 1`.
            if i + 1 >= self.buf.len() {
                break;
            }
            let frac = (self.phase - i as f64) as f32;
            let s = self.buf[i] + (self.buf[i + 1] - self.buf[i]) * frac;
            out.push(s);
            self.phase += self.step;
        }
        // Drop fully-consumed input, retaining one sample for continuity.
        let consumed = self.phase.floor() as usize;
        let drop = consumed.saturating_sub(1);
        if drop > 0 {
            self.buf.drain(..drop);
            self.phase -= drop as f64;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::LinearResampler;

    #[test]
    fn upsample_16k_to_48k_is_3x() {
        let mut r = LinearResampler::new(16000, 48000);
        let input = vec![0.0f32; 16000]; // 1 second at 16 kHz
        let out = r.push(&input);
        let expected = 48000;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(diff < 100, "expected ~{expected} samples, got {}", out.len());
    }

    #[test]
    fn downsample_48k_to_16k_is_third() {
        let mut r = LinearResampler::new(48000, 16000);
        let input = vec![0.0f32; 48000]; // 1 second at 48 kHz
        let out = r.push(&input);
        let expected = 16000;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(diff < 100, "expected ~{expected} samples, got {}", out.len());
    }

    #[test]
    fn upsample_8k_to_16k_is_2x() {
        let mut r = LinearResampler::new(8000, 16000);
        let input = vec![0.0f32; 8000]; // 1 second at 8 kHz
        let out = r.push(&input);
        let expected = 16000;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(diff < 100, "expected ~{expected} samples, got {}", out.len());
    }
}
