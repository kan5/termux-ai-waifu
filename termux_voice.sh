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
# 4s keeps the loop snappy — speak a short phrase, get a quick reply.
RECORD_LIMIT="${RECORD_LIMIT:-4}"

record() {
    local t0 t1
    t0=$SECONDS
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
    t1=$SECONDS
    echo "  [замер] запись: $((t1-t0))с"
}

run_offline() {
    local t0 t1 t2 t3
    t0=$SECONDS
    echo "▷ Декодирование в mono 16k WAV (с усилением)..."
    # -af volume=6dB: phone mics are quiet; a modest boost so Silero VAD
    # detects speech, without over-amplifying noise/echo into false triggers.
    ffmpeg -y -v error -i "$RAW" -ar 16000 -ac 1 -af "volume=6dB" "$IN"
    t1=$SECONDS
    echo "  [замер] ffmpeg декод: $((t1-t0))с"

    echo "▷ Распознавание + ответ (VAD→STT→LLM→TTS)..."
    # TTS-сервис должен быть запущен (./setup_termux.sh tts; python tts_service/service.py ...)
    # LD_LIBRARY_PATH на локальный каталог: бинарь использует NDK libc++_shared.so
    # и libonnxruntime.so, положенные рядом (см. build_termux_cross.sh).
    LD_LIBRARY_PATH="$(pwd)" ./voice-assistant-android \
        --config config.termux.toml \
        --file-input "$IN" \
        --file-output "$OUT"
    t2=$SECONDS
    echo "  [замер] бинарь (STT+LLM+TTS): $((t2-t1))с"

    echo "▶ Проигрывание ответа..."
    # Play only if the reply was actually produced this round (it's skipped when
    # the previous step errored, e.g. no speech detected).
    if [[ -s "$OUT" ]]; then
        # termux-media-player plays async and returns immediately, so wait for
        # the audio to finish before we start recording again — otherwise the
        # mic hears our own reply and triggers an echo loop.
        local dur
        dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nokey=1 "$OUT" 2>/dev/null || echo 0)
        termux-media-player play "$OUT"
        # Give the player a moment to start, then wait out the audio length.
        sleep 0.5
        sleep "${dur%.*}"
        echo "  [замер] воспроизведение: ~${dur%.*}с"
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
