#!/usr/bin/env bash
# === Voice Assistant — сборка и запуск в Termux (Android) ===
# Ставит все зависимости через pkg (НЕ через pip для torch/numpy),
# скачивает модели и собирает Rust-бинарный файл прямо на устройстве.
#
# Использование:
#   ./setup_termux.sh build       # установить зависимости + собрать бинарь
#   ./setup_termux.sh models      # только скачать модели
#   ./setup_termux.sh tts         # только установить python TTS-зависимости
#   ./setup_termux.sh run         # собрать + запустить ассистента
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Обновление Termux и установка базовых пакетов"
pkg update -y
pkg upgrade -y

# --- Тулчейн для Rust-бинаря -----------------------------------------
# rust собирает под aarch64-linux-android; llama.cpp / transcribe.cpp /
# ort компилируются из исходников через CMake прямо на устройстве.
echo "==> Установка Rust-тулчейна и компиляторов"
pkg install -y rust cmake clang make ninja pkg-config binutils \
    onnxruntime curl

# --- Android-аудио (Termux:API + ffmpeg) ------------------------------
# termux-microphone-record не пишет WAV/PCM (только m4a/amr/opus), поэтому
# запись декодируется в mono 16k WAV через ffmpeg. Воспроизведение —
# через termux-media-player (входит в termux-api).
echo "==> Установка Termux:API и ffmpeg для аудио-I/O"
pkg install -y termux-api ffmpeg

# --- Python-сторона (TTS-сервис) --------------------------------------
# torch и numpy ставим из репозитория Termux (предсобранный CPU-вариант),
# НЕ через pip. robyn в Termux отсутствует, поэтому используется
# встроенный stdlib-сервер (см. tts_service/service.py).
echo "==> Установка Python + PyTorch (pkg, не pip)"
pkg install -y python python-torch python-numpy

build() {
    echo "==> Сборка Rust-бинаря (cargo build --release)"
    # ONNX Runtime: используем системный libonnxruntime.so из пакета onnxruntime.
    export ORT_LIB_LOCATION="${PREFIX:-/data/data/com.termux/files/usr}/lib/libonnxruntime.so"
    cargo build --release
    echo "OK: target/release/voice-assistant"
}

models() {
    echo "==> Скачивание моделей"
    ./download_models.sh all
}

tts() {
    echo "==> Python TTS-сервис готов (зависимости уже стоят через pkg)"
    echo "    Запуск: python tts_service/service.py --host 127.0.0.1 --port 8090"
}

case "${1:-build}" in
    build)  build ;;
    models) models ;;
    tts)    tts ;;
    run)
        build
        models
        echo "==> Запуск TTS-сервиса и ассистента"
        python tts_service/service.py --host 127.0.0.1 --port 8090 &
        exec ./target/release/voice-assistant --config config.toml
        ;;
    *)
        echo "unknown target: $1 (use build|models|tts|run)" >&2
        exit 1
        ;;
esac
