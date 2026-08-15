#!/usr/bin/env python3
"""Silero TTS service (v5_5_ru) behind Robyn.

Exposes a single endpoint:

    POST /tts
    Content-Type: application/json
    Body: {"text": "...", "speaker": "xenia"}

    -> 200, Content-Type: application/octet-stream
       body = raw little-endian f32 PCM, mono, 16 kHz

The Rust application POSTs text and plays back the returned PCM directly.

Run:
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python service.py --host 127.0.0.1 --port 8090

The first run downloads the Silero model from torch.hub (cached afterwards).
"""

import argparse

import numpy as np
import torch
from robyn import Robyn, Response

app = Robyn(__file__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The v5_5_ru model natively produces 48 kHz and supports output rates
# 8000/24000/48000. We generate at 8 kHz (fewest vocoder samples -> faster
# time-to-first-sound) and upsample to the pipeline's internal 16 kHz.
MODEL_RATE = 8000
TARGET_RATE = 16000


def load_model():
    # v5_5_ru is the Russian TTS model required by the ТЗ.
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v5_5_ru",
        trust_repo=True,
    )
    model.to(DEVICE)
    return model


def resample_linear(audio: np.ndarray, orig_rate: int, new_rate: int) -> np.ndarray:
    """Linear-interpolation resample for a 1-D mono signal."""
    if orig_rate == new_rate:
        return audio
    n_out = int(audio.shape[0] * new_rate / orig_rate)
    x = np.linspace(0.0, audio.shape[0] - 1.0, n_out)
    x0 = np.floor(x).astype(np.int64)
    x1 = np.minimum(x0 + 1, audio.shape[0] - 1)
    frac = (x - x0).astype(np.float32)
    return audio[x0] * (1.0 - frac) + audio[x1] * frac


MODEL = load_model()


@app.post("/tts")
async def tts(request):
    data = request.json()
    text = data.get("text", "").strip()
    speaker = data.get("speaker", "xenia")
    if not text:
        return Response(
            status_code=400,
            headers={"Content-Type": "text/plain"},
            body=b"empty text",
        )

    audio = MODEL.apply_tts(text=text, speaker=speaker, sample_rate=MODEL_RATE)
    samples = audio.detach().cpu().numpy()
    samples = resample_linear(samples, MODEL_RATE, TARGET_RATE)
    pcm = samples.astype("<f4").tobytes()
    return Response(
        status_code=200,
        headers={"Content-Type": "application/octet-stream"},
        body=pcm,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    app.start(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
