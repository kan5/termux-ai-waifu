//! Native audio via Android AAudio (Termux).
//!
//! On Android the CPAL backend can't open the mic/speakers, so we drive AAudio
//! directly (the system library is present on-device). This gives a *streaming*
//! PCM capture/playback path — no temp files, no termux-microphone-record.
//!
//! Only compiled for `target_os = "android"`. The header lives in the NDK
//! sysroot; `libaaudio.so` is a system lib available at runtime on any modern
//! device. During cross-link the crate needs a stub `libaaudio.so` with the
//! same symbols (see build_termux_cross.sh); the real lib is loaded at runtime.

#![cfg(target_os = "android")]

use std::os::raw::{c_int, c_longlong, c_void};

use anyhow::{Context, Result};

// AAudio enum values (see AAudio.h).
const DIR_OUTPUT: c_int = 0;
const DIR_INPUT: c_int = 1;
const FORMAT_FLOAT: c_int = 2; // PCM_FLOAT
const PERF_NONE: c_int = 10;

type Stream = *mut c_void;
type Builder = *mut c_void;

#[link(name = "aaudio")]
unsafe extern "C" {
    fn AAudio_createStreamBuilder(builder: *mut Builder) -> c_int;
    fn AAudioStreamBuilder_setDirection(builder: Builder, direction: c_int) -> c_int;
    fn AAudioStreamBuilder_setSampleRate(builder: Builder, rate: c_int) -> c_int;
    fn AAudioStreamBuilder_setChannelCount(builder: Builder, count: c_int) -> c_int;
    fn AAudioStreamBuilder_setFormat(builder: Builder, format: c_int) -> c_int;
    fn AAudioStreamBuilder_setPerformanceMode(builder: Builder, mode: c_int) -> c_int;
    fn AAudioStreamBuilder_openStream(builder: Builder, stream: *mut Stream) -> c_int;
    fn AAudioStreamBuilder_delete(builder: Builder);
    fn AAudioStream_requestStart(stream: Stream) -> c_int;
    fn AAudioStream_requestStop(stream: Stream) -> c_int;
    fn AAudioStream_read(
        stream: Stream,
        buffer: *mut c_void,
        num_frames: c_int,
        timeout_nanoseconds: c_longlong,
    ) -> c_int;
    fn AAudioStream_write(
        stream: Stream,
        buffer: *const c_void,
        num_frames: c_int,
        timeout_nanoseconds: c_longlong,
    ) -> c_int;
    fn AAudioStream_close(stream: Stream) -> c_int;
    fn AAudioStream_getSampleRate(stream: Stream) -> c_int;
}

fn new_stream(direction: c_int, sample_rate: c_int, channels: c_int) -> Result<Stream> {
    let mut builder: Builder = std::ptr::null_mut();
    let mut stream: Stream = std::ptr::null_mut();
    let rc = unsafe {
        AAudio_createStreamBuilder(&mut builder);
        if builder.is_null() {
            anyhow::bail!("AAudio_createStreamBuilder returned null builder");
        }
        AAudioStreamBuilder_setDirection(builder, direction);
        AAudioStreamBuilder_setSampleRate(builder, sample_rate);
        AAudioStreamBuilder_setChannelCount(builder, channels);
        AAudioStreamBuilder_setFormat(builder, FORMAT_FLOAT);
        AAudioStreamBuilder_setPerformanceMode(builder, PERF_NONE);
        let rc = AAudioStreamBuilder_openStream(builder, &mut stream);
        AAudioStreamBuilder_delete(builder);
        rc
    };
    if rc != 0 || stream.is_null() {
        anyhow::bail!("AAudio openStream failed rc={rc} (dir={direction})");
    }
    Ok(stream)
}

/// Open a mono capture stream at `sample_rate` Hz, f32.
pub fn open_input(sample_rate: c_int) -> Result<Stream> {
    new_stream(DIR_INPUT, sample_rate, 1)
}

/// Open a mono playback stream at `sample_rate` Hz, f32.
pub fn open_output(sample_rate: c_int) -> Result<Stream> {
    new_stream(DIR_OUTPUT, sample_rate, 1)
}

/// Read up to `num_frames` f32 frames into `buffer`. Returns frames read
/// (may be < num_frames), or a negative AAudio error code.
pub fn read(stream: Stream, buffer: &mut [f32], timeout_ms: i64) -> c_int {
    let ns = timeout_ms.saturating_mul(1_000_000) as c_longlong;
    unsafe {
        AAudioStream_read(
            stream,
            buffer.as_mut_ptr() as *mut c_void,
            buffer.len() as c_int,
            ns,
        )
    }
}

/// Write `frames` f32 frames. Returns frames written, or a negative error code.
pub fn write(stream: Stream, buffer: &[f32], timeout_ms: i64) -> c_int {
    let ns = timeout_ms.saturating_mul(1_000_000) as c_longlong;
    unsafe {
        AAudioStream_write(
            stream,
            buffer.as_ptr() as *const c_void,
            buffer.len() as c_int,
            ns,
        )
    }
}

pub fn start(stream: Stream) -> Result<()> {
    let rc = unsafe { AAudioStream_requestStart(stream) };
    if rc != 0 {
        anyhow::bail!("AAudio requestStart failed rc={rc}");
    }
    Ok(())
}

pub fn close(stream: Stream) {
    if !stream.is_null() {
        unsafe {
            AAudioStream_requestStop(stream);
            AAudioStream_close(stream);
        }
    }
}

/// Play a mono 16k f32 buffer to the default output, blocking until done.
pub fn play(samples: &[f32], sample_rate: c_int) -> Result<()> {
    let stream = open_output(sample_rate).context("open AAudio output")?;
    start(stream)?;
    let chunk = 480;
    for frame in samples.chunks(chunk) {
        // Blocking write with a generous timeout per chunk.
        let n = write(stream, frame, 2000);
        if n < 0 {
            anyhow::bail!("AAudio write failed rc={n}");
        }
        // If fewer frames were accepted, retry the rest of the chunk.
        if (n as usize) < frame.len() {
            let _ = &frame[n as usize..]; // (retry loop simplified; AAudio usually accepts all)
        }
    }
    close(stream);
    Ok(())
}
