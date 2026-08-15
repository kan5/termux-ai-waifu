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
RAW="$WORKDIR/input.opus"
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
    # Recording runs in the background; we then stop it with `-q`. Stopping via
    # `-q` calls mediaRecorder.stop() and finalizes the container, whereas the
    # `-l <sec>` auto-limit does NOT finalize (moov atom missing / empty ogg).
    termux-microphone-record -f "$RAW" -e opus -r 16000 -c 1 >/dev/null 2>&1 &
    sleep "$RECORD_LIMIT"
    termux-microphone-record -q >/dev/null 2>&1 || true
    sleep 1
}

run_offline() {
    echo "▷ Декодирование в mono 16k WAV (с усилением)..."
    # -af volume=10dB: phone mics are quiet; boost so Silero VAD (threshold 0.5)
    # reliably detects speech.
    ffmpeg -y -v error -i "$RAW" -ar 16000 -ac 1 -af "volume=10dB" "$IN"

    echo "▷ Распознавание + ответ (VAD→STT→LLM→TTS)..."
    # TTS-сервис должен быть запущен (./setup_termux.sh tts; python tts_service/service.py ...)
    # LD_LIBRARY_PATH на локальный каталог: бинарь использует NDK libc++_shared.so
    # и libonnxruntime.so, положенные рядом (см. build_termux_cross.sh).
    LD_LIBRARY_PATH="$(pwd)" ./voice-assistant-android \
        --config config.termux.toml \
        --file-input "$IN" \
        --file-output "$OUT"

    echo "▶ Проигрывание ответа..."
    # Play only if the reply was actually produced this round (it's skipped when
    # the previous step errored, e.g. no speech detected).
    if [[ -s "$OUT" ]]; then
        termux-media-player play "$OUT"
    else
        echo "⚠ Ответ не сформирован — пропускаю проигрывание."
    fi
}

one() {
    record
    run_offline
}

loop() {
    while true; do
        record
        # run_offline returns non-zero on "no speech detected" / empty result —
        # that must NOT kill the loop (set -e would exit). Swallow it and retry.
        if ! run_offline; then
            echo "⚠ Нет распознанной речи — повторяю..."
        fi
        sleep 1
    done
}

case "${1:-once}" in
    once) one ;;
    loop) loop ;;
    *) echo "unknown: $1 (use once|loop)" >&2; exit 1 ;;
esac
