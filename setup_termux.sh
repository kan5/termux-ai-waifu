#!/usr/bin/env bash
# === Voice Assistant — настройка Termux (Android) ===
#
# Ставит зависимости через pkg (НЕ через pip для torch/numpy).
# Rust-бинарный файл здесь НЕ собирается: llama-cpp-2 требует Android NDK,
# которого в Termux нет. Вместо этого бинарь кросс-компилируется на десктопе
# (см. build_termux_cross.sh) и копируется в Termux.
#
# Использование:
#   ./setup_termux.sh deps        # установить зависимости (python, torch, onnxruntime)
#   ./setup_termux.sh models      # скачать модели
#   ./setup_termux.sh tts         # запустить Python TTS-сервис (зависимости уже стоят)
#   ./setup_termux.sh run         # показать, как запустить ассистента
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Обновление Termux"
pkg update -y
pkg upgrade -y

# --- Базовые пакеты ------------------------------------------------
# onnxruntime даёт системный libonnxruntime.so — бинарь dlopen'ит его в рантайме.
# curl — для скачивания моделей.
echo "==> Установка базовых пакетов (onnxruntime, curl)"
pkg install -y onnxruntime curl

# --- Python-сторона (TTS-сервис) -----------------------------------
# torch и numpy ставим из репозитория Termux (предсобранный CPU-вариант),
# НЕ через pip. robyn в Termux отсутствует, поэтому используется
# встроенный stdlib-сервер (см. tts_service/service.py).
echo "==> Установка Python + PyTorch (pkg, не pip)"
pkg install -y python python-torch python-numpy

deps() {
    echo "OK: зависимости установлены. Бинарь собирается на десктопе:"
    echo "    ./build_termux_cross.sh   (см. README)"
}

models() {
    echo "==> Скачивание моделей"
    ./download_models.sh all
    # Для Termux — быстрая non-reasoning модель LLM.
    if [[ ! -f models/qwen2.5-0.5b-instruct-q4_k_m.gguf ]]; then
        echo "==> Скачивание Qwen2.5-0.5B-Instruct (быстрая модель для Termux)"
        curl -L --fail --progress-bar \
            -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    fi
    echo "OK: модели в models/"
}

tts() {
    echo "==> Запуск Python TTS-сервиса (127.0.0.1:8090)"
    echo "    (torch/numpy уже стоят через pkg, venv не нужен)"
    exec python tts_service/service.py --host 127.0.0.1 --port 8090
}

run() {
    echo "Бинарь соберите на десктопе (./build_termux_cross.sh) и скопируйте в Termux."
    echo "Затем:"
    echo "  1) TTS-сервис:      python tts_service/service.py --host 127.0.0.1 --port 8090"
    echo "  2) потоковый режим: cd ~/va && LD_LIBRARY_PATH=\"\$(pwd)\" ./voice-assistant --config config.termux.toml"
}

case "${1:-deps}" in
    deps)  deps ;;
    models) models ;;
    tts)   tts ;;
    run)   run ;;
    *)
        echo "unknown target: $1 (use deps|models|tts|run)" >&2
        exit 1
        ;;
esac
