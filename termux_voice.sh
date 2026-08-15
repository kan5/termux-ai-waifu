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
# Auto-stop recording after this many seconds (0 = unlimited, stop with Ctrl-C).
# A fixed limit makes the loop hands-free: speak, wait, hear the reply.
RECORD_LIMIT="${RECORD_LIMIT:-8}"

record() {
    echo "▶ Говорите $RECORD_LIMIT с..."
    # termux-microphone-record refuses to start if the file already exists, and
    # the loop reuses the same names — remove stale files first. Also stop any
    # previous recording (in case the loop was interrupted mid-record).
    termux-microphone-record -q 2>/dev/null || true
    rm -f "$RAW" "$IN" "$OUT"
    # termux-microphone-record writes compressed m4a (no WAV/PCM support); we
    # decode to mono 16k WAV afterwards. -l auto-stops after RECORD_LIMIT sec.
    termux-microphone-record -f "$RAW" -e aac -r 16000 -c 1 -l "$RECORD_LIMIT"
}

run_offline() {
    echo "▷ Декодирование в mono 16k WAV..."
    ffmpeg -y -v error -i "$RAW" -ar 16000 -ac 1 "$IN"

    echo "▷ Распознавание + ответ (VAD→STT→LLM→TTS)..."
    # TTS-сервис должен быть запущен (./setup_termux.sh tts; python tts_service/service.py ...)
    # LD_LIBRARY_PATH на локальный каталог: бинарь использует NDK libc++_shared.so
    # и libonnxruntime.so, положенные рядом (см. build_termux_cross.sh).
    LD_LIBRARY_PATH="$(pwd)" ./voice-assistant-android \
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
