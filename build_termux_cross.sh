#!/usr/bin/env bash
# === Cross-compile voice-assistant for Android/Termux (on a desktop Linux) ===
#
# Builds an aarch64 ELF executable that runs in Termux. llama-cpp-sys-2 /
# transcribe-cpp build natively via the Android NDK; ort uses load-dynamic
# (dlopen's a system libonnxruntime.so at runtime).
#
# Requirements:
#   rustup target add aarch64-linux-android
#   cargo install cargo-ndk
#   Android NDK (e.g. r27c) somewhere; point ANDROID_NDK_ROOT at it.
#
# Usage:
#   ./build_termux_cross.sh [--ndk /path/to/android-ndk-*]
set -euo pipefail

cd "$(dirname "$0")"

# --- NDK ---
if [[ -n "${ANDROID_NDK_ROOT:-}" ]]; then
    NDK="$ANDROID_NDK_ROOT"
else
    for d in "$HOME/Android/Sdk/android-ndk-"* /opt/android-ndk* /usr/lib/android-ndk; do
        if [[ -d "$d" ]]; then NDK="$d"; break; fi
    done
fi
[[ -n "${NDK:-}" ]] || { echo "Android NDK not found (set ANDROID_NDK_ROOT)"; exit 1; }
export ANDROID_NDK_ROOT="$NDK"
export ANDROID_NDK_HOME="$NDK"

# API level: must be >= 23 for bionic to expose POSIX_MADV_* (llama.cpp uses it).
# The llama-cpp-sys build.rs does NOT forward cargo-ndk's -P to its own CMake, so
# we pass the API level via env ANDROID_API_LEVEL/ANDROID_PLATFORM too.
export ANDROID_API_LEVEL=24
export ANDROID_PLATFORM=android-24
export CARGO_NDK_ANDROID_PLATFORM=24

# --- pthread stub ---
# transcribe-cpp-sys emits cargo:rustc-link-lib=dylib=pthread. On bionic pthread
# is part of libc (no libpthread.so), so the NDK link fails. Provide an empty
# aarch64 libpthread.so to satisfy the -lpthread flag.
STUB_DIR="$(mktemp -d)"
cat > "$STUB_DIR/pthread_stub.c" <<'EOF'
/* Empty: on bionic pthread lives in libc. */
EOF
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_API_LEVEL}-clang" \
    -shared -o "$STUB_DIR/libpthread.so" "$STUB_DIR/pthread_stub.c"

export RUSTFLAGS="-L $STUB_DIR ${RUSTFLAGS:-}"

echo "==> Cross-compiling for aarch64-linux-android (API $ANDROID_API_LEVEL)..."
cargo ndk -t arm64-v8a -P "$ANDROID_API_LEVEL" build --release
# cargo-ndk exits with "No usable artifacts" for a plain bin crate; the binary
# is still produced at target/aarch64-linux-android/release/voice-assistant.
BIN="target/aarch64-linux-android/release/voice-assistant"
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: expected binary not found: $BIN" >&2
    exit 1
fi
echo "==> OK: $BIN ($(du -h "$BIN" | cut -f1))"
echo "    Copy it to Termux and run with --file-input / --file-output."
