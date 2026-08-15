//! Audio capture and playback via CPAL.
//!
//! Converts between the device's native format and the internal
//! mono / 16 kHz / f32 representation (ТЗ §4). Sample-rate conversion is
//! linear interpolation; channel downmix averages all channels.

mod resample;
pub use resample::LinearResampler;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use tokio::sync::mpsc;

use crate::config::AudioConfig;
use crate::traits::{AudioCapture, AudioSink};
use crate::types::{AudioChunk, PlaybackCommand};

/// Microphone capture → mono 16 kHz f32 chunks.
pub struct CpalCapture {
    config: AudioConfig,
}

impl CpalCapture {
    pub fn new(config: AudioConfig) -> Self {
        Self { config }
    }
}

impl AudioCapture for CpalCapture {
    fn stream(self: Box<Self>, tx: mpsc::Sender<AudioChunk>) -> Result<()> {
        let host = cpal::default_host();
        let device = find_input_device(&host, self.config.input_device.as_deref())?;
        let supported = device
            .default_input_config()
            .context("no default input config")?;
        let format = supported.sample_format();
        let channels = supported.channels() as usize;
        let device_rate = supported.sample_rate();
        let stream_config = supported.config();
        tracing::info!(?format, channels, device_rate, "opening input stream");

        match format {
            SampleFormat::I16 => run_input::<i16>(&device, stream_config, channels, device_rate, &self.config, tx),
            SampleFormat::U16 => run_input::<u16>(&device, stream_config, channels, device_rate, &self.config, tx),
            SampleFormat::F32 => run_input::<f32>(&device, stream_config, channels, device_rate, &self.config, tx),
            other => Err(anyhow!("unsupported input sample format: {other:?}")),
        }
    }
}

/// Audio playback → output device.
pub struct CpalSink {
    config: AudioConfig,
}

impl CpalSink {
    pub fn new(config: AudioConfig) -> Self {
        Self { config }
    }
}

impl AudioSink for CpalSink {
    fn play(self: Box<Self>, mut rx: mpsc::Receiver<PlaybackCommand>) -> Result<()> {
        let host = cpal::default_host();
        let device = find_output_device(&host, self.config.output_device.as_deref())?;
        let supported = device
            .default_output_config()
            .context("no default output config")?;
        let format = supported.sample_format();
        let channels = supported.channels() as usize;
        let device_rate = supported.sample_rate();
        let stream_config = supported.config();
        tracing::info!(?format, channels, device_rate, "opening output stream");

        match format {
            SampleFormat::I16 => run_output::<i16>(&device, stream_config, channels, device_rate, &self.config, &mut rx),
            SampleFormat::U16 => run_output::<u16>(&device, stream_config, channels, device_rate, &self.config, &mut rx),
            SampleFormat::F32 => run_output::<f32>(&device, stream_config, channels, device_rate, &self.config, &mut rx),
            other => Err(anyhow!("unsupported output sample format: {other:?}")),
        }
    }
}

fn find_input_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device> {
    match name {
        Some(name) => host
            .input_devices()
            .context("no input devices")?
            .find(|d| d.description().map(|desc| desc.name() == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("input device not found: {name}")),
        None => host.default_input_device().context("no default input device"),
    }
}

fn find_output_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device> {
    match name {
        Some(name) => host
            .output_devices()
            .context("no output devices")?
            .find(|d| d.description().map(|desc| desc.name() == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("output device not found: {name}")),
        None => host.default_output_device().context("no default output device"),
    }
}

/// Convert interleaved device samples to mono f32 (average across channels).
fn samples_to_mono_f32<T>(data: &[T], channels: usize) -> Vec<f32>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let frames = data.len() / channels;
    let mut out = Vec::with_capacity(frames);
    for frame in data.chunks_exact(channels) {
        let mut sum = 0.0f32;
        for &s in frame {
            sum += f32::from_sample(s);
        }
        out.push(sum / channels as f32);
    }
    out
}

fn run_input<T>(
    device: &cpal::Device,
    stream_config: cpal::StreamConfig,
    channels: usize,
    device_rate: u32,
    acfg: &AudioConfig,
    tx: mpsc::Sender<AudioChunk>,
) -> Result<()>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let target_rate = acfg.input_sample_rate.unwrap_or(device_rate);
    let mut stream_config = stream_config;
    if target_rate != device_rate {
        tracing::info!(reported = device_rate, target = target_rate, "overriding input sample rate");
    }
    stream_config.sample_rate = target_rate;
    let mut resampler = LinearResampler::new(target_rate, acfg.sample_rate);
    let chunk_size = acfg.chunk_samples;
    let mut out_buf: Vec<f32> = Vec::new();
    let tx_check = tx.clone();

    let stream = device
        .build_input_stream::<T, _, _>(
            stream_config,
            move |data: &[T], _| {
                let mono = samples_to_mono_f32(data, channels);
                out_buf.extend_from_slice(&resampler.push(&mono));
                while out_buf.len() >= chunk_size {
                    let piece: Vec<f32> = out_buf.drain(..chunk_size).collect();
                    if tx.try_send(AudioChunk::new(piece)).is_err() {
                        tracing::warn!("input channel full or closed; dropping chunk");
                    }
                }
            },
            |err| tracing::error!("input stream error: {err}"),
            None,
        )
        .context("failed to build input stream")?;

    stream.play().context("failed to start input stream")?;

    // Keep the stream alive until the receiver side is dropped (shutdown).
    while !tx_check.is_closed() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Ok(())
}

fn run_output<T>(
    device: &cpal::Device,
    stream_config: cpal::StreamConfig,
    channels: usize,
    device_rate: u32,
    acfg: &AudioConfig,
    rx: &mut mpsc::Receiver<PlaybackCommand>,
) -> Result<()>
where
    T: SizedSample + FromSample<f32>,
{
    let queue: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
    let cb_queue = Arc::clone(&queue);
    let target_rate = acfg.output_sample_rate.unwrap_or(device_rate);
    let mut stream_config = stream_config;
    if target_rate != device_rate {
        tracing::info!(reported = device_rate, target = target_rate, "overriding output sample rate");
    }
    stream_config.sample_rate = target_rate;
    let mut resampler = LinearResampler::new(acfg.sample_rate, target_rate);

    let stream = device
        .build_output_stream::<T, _, _>(
            stream_config,
            move |data: &mut [T], _| {
                let mut q = cb_queue.lock().unwrap();
                let frames = data.len() / channels;
                let mut idx = 0;
                for _ in 0..frames {
                    let s = q.pop_front().unwrap_or(0.0f32);
                    for _ in 0..channels {
                        data[idx] = T::from_sample(s);
                        idx += 1;
                    }
                }
            },
            |err| tracing::error!("output stream error: {err}"),
            None,
        )
        .context("failed to build output stream")?;

    stream.play().context("failed to start output stream")?;

    // Producer loop: feed resampled audio into the playback queue.
    while let Some(cmd) = rx.blocking_recv() {
        match cmd {
            PlaybackCommand::Play(chunk) => {
                let out = resampler.push(chunk.as_slice());
                queue.lock().unwrap().extend(out);
            }
            PlaybackCommand::Flush => {
                queue.lock().unwrap().clear();
            }
        }
    }

    Ok(())
}
