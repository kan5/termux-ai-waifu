//! LLM backend.
//!
//! Two implementations of the [`crate::traits::Llm`] trait:
//!
//! - **Desktop** (`QwenLlm`, llama.cpp via the `llama-cpp-2` crate): the crate
//!   compiles fine on Linux/macOS.
//! - **Android/Termux** (`NativeLlm`): `llama-cpp-2`'s build.rs requires the
//!   Android NDK on `aarch64-linux-android`, which Termux lacks. So on Android
//!   we instead dlopen the system `libllama.so` (from the `llama-cpp` pkg) and
//!   drive it directly. See `native.rs`.

#[cfg(not(target_os = "android"))]
mod qwen;
#[cfg(not(target_os = "android"))]
pub use qwen::QwenLlm;

#[cfg(target_os = "android")]
mod native;
#[cfg(target_os = "android")]
pub use native::NativeLlm;
