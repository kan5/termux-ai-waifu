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

# --- aaudio stub ---
# libaaudio.so is a system Android lib (present at runtime) but is NOT shipped
# in the NDK aarch64 sysroot, so we can't link against it. Provide a stub .so
# with the AAudio entry points we use; the real system lib is loaded at runtime.
cat > "$STUB_DIR/aaudio_stub.c" <<'EOF'
typedef void AAudioStream; typedef void AAudioStreamBuilder;
int AAudio_createStreamBuilder(AAudioStreamBuilder **b){ return 0; }
int AAudioStreamBuilder_setDirection(AAudioStreamBuilder *b, int d){ return 0; }
int AAudioStreamBuilder_setSampleRate(AAudioStreamBuilder *b, int r){ return 0; }
int AAudioStreamBuilder_setChannelCount(AAudioStreamBuilder *b, int c){ return 0; }
int AAudioStreamBuilder_setFormat(AAudioStreamBuilder *b, int f){ return 0; }
int AAudioStreamBuilder_setPerformanceMode(AAudioStreamBuilder *b, int m){ return 0; }
int AAudioStreamBuilder_openStream(AAudioStreamBuilder *b, AAudioStream **s){ return 0; }
int AAudioStreamBuilder_delete(AAudioStreamBuilder *b){ return 0; }
int AAudioStream_requestStart(AAudioStream *s){ return 0; }
int AAudioStream_requestStop(AAudioStream *s){ return 0; }
int AAudioStream_read(AAudioStream *s, void *b, int n, long long t){ return 0; }
int AAudioStream_write(AAudioStream *s, const void *b, int n, long long t){ return 0; }
int AAudioStream_close(AAudioStream *s){ return 0; }
int AAudioStream_getSampleRate(AAudioStream *s){ return 0; }
EOF
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_API_LEVEL}-clang" \
    -shared -o "$STUB_DIR/libaaudio.so" "$STUB_DIR/aaudio_stub.c"

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

# The binary links against libc++_shared.so; Termux ships an older libc++ whose
# ABI lacks some symbols the NDK-r27 build references. Copy the NDK's own
# libc++_shared.so next to the binary and run with LD_LIBRARY_PATH pointing at
# it (see README).
LIBDIR="$(dirname "$BIN")"
cp "$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
    "$LIBDIR/libc++_shared.so"

echo "==> OK: $BIN ($(du -h "$BIN" | cut -f1))"
echo "    + libc++_shared.so copied alongside (needed on Termux)"
echo "    Copy both to Termux and run with --config config.termux.toml."
