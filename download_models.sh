#!/usr/bin/env bash
# Download the MVP models into ./models.
# Usage: ./download_models.sh [vad|stt|llm|all]
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p models

download() {
    local url="$1" out="$2"
    if [[ -f "$out" ]]; then
        echo "skip (exists): $out"
        return
    fi
    echo "downloading: $out"
    curl -L --fail --progress-bar -o "$out" "$url"
    echo "ok: $out"
}

VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
VAD_OUT="models/silero_vad.onnx"

STT_URL="https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q8_0.gguf"
STT_OUT="models/gigaam-v3-e2e-ctc-Q8_0.gguf"

LLM_URL="https://huggingface.co/Mungert/Qwen3-0.6B-abliterated-GGUF/resolve/main/Qwen3-0.6B-abliterated-q4_k_m.gguf"
LLM_OUT="models/Qwen3-0.6B-abliterated-q4_k_m.gguf"

case "${1:-all}" in
    vad) download "$VAD_URL" "$VAD_OUT" ;;
    stt) download "$STT_URL" "$STT_OUT" ;;
    llm) download "$LLM_URL" "$LLM_OUT" ;;
    all)
        download "$VAD_URL" "$VAD_OUT"
        download "$STT_URL" "$STT_OUT"
        download "$LLM_URL" "$LLM_OUT"
        ;;
    *)
        echo "unknown target: $1 (use vad|stt|llm|all)" >&2
        exit 1
        ;;
esac
