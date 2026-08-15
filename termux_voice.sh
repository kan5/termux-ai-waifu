#!/usr/bin/env bash
# === Одно-репликавый цикл ассистента на Termux (file-режим) ===
#
#   запись (termux-microphone-record) → ffmpeg в mono 16k WAV →
#   voice-assistant --file-input/--file-output → воспроизведение
#   (termux-media-player).
#
# Использование:
#   ./termux_voice.sh            # записать реплику, ответить, проиграть
#   ./termux_voice.sh once       # то же самое (однократно)
#   ./termux_voice.sh loop       # бесконечный цикл запись→ответ→воспроизведение
set -euo pipefail

cd "$(dirname "$0")"

WORKDIR="${WORKDIR:-/data/data/com.termux/files/home/va_tmp}"
mkdir -p "$WORKDIR"
RAW="$WORKDIR/input.m4a"
IN="$WORKDIR/input.wav"
OUT="$WORKDIR/output.wav"

record() {
    echo "▶ Говорите (Ctrl-C / второй Ctrl-C, чтобы остановить запись)..."
    # Компактный формат m4a (termux-microphone-record не пишет WAV/PCM).
    termux-microphone-record -f "$RAW" -e aac -r 16000 -c 1
}

run_offline() {
    echo "▷ Декодирование в mono 16k WAV..."
    ffmpeg -y -v error -i "$RAW" -ar 16000 -ac 1 "$IN"

    echo "▷ Распознавание + ответ (VAD→STT→LLM→TTS)..."
    # TTS-сервис должен быть запущен (./setup_termux.sh tts; python tts_service/service.py ...)
    ./target/release/voice-assistant \
        --config config.toml \
        --file-input "$IN" \
        --file-output "$OUT"

    echo "▶ Проигрывание ответа..."
    termux-media-player play "$OUT"
}

one() {
    record
    run_offline
}

loop() {
    while true; do
        record
        run_offline
    done
}

case "${1:-once}" in
    once) one ;;
    loop) loop ;;
    *) echo "unknown: $1 (use once|loop)" >&2; exit 1 ;;
esac
