#!/usr/bin/env python3
"""Silero TTS service (v5_5_ru) over stdlib ThreadingHTTPServer.

Exposes a single endpoint:

    POST /tts
    Content-Type: application/json
    Body: {"text": "...", "speaker": "xenia"}

    -> 200, Content-Type: application/octet-stream
       body = raw little-endian f32 PCM, mono, 16 kHz

The Rust application POSTs text and plays back the returned PCM directly.

The server uses the stdlib (ThreadingHTTPServer) so it runs anywhere with a
system Python + torch/numpy — including Termux, where those come from `pkg`
(python-torch / python-numpy) and there is no need for pip at all.

Run:
    python service.py --host 127.0.0.1 --port 8090

The first run downloads the Silero model from torch.hub (cached afterwards).
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

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


class TtsHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        print("[%s] %s" % (self.log_date_time_string(), format % args), flush=True)

    def do_POST(self):
        if self.path != "/tts":
            self.send_error(404, "not found")
            return

        # Read the JSON body.
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8"))
            text = (data.get("text") or "").strip()
            speaker = data.get("speaker") or "xenia"
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._respond(400, b"bad json", "text/plain")
            return

        if not text:
            self._respond(400, b"empty text", "text/plain")
            return

        audio = MODEL.apply_tts(text=text, speaker=speaker, sample_rate=MODEL_RATE)
        samples = audio.detach().cpu().numpy()
        samples = resample_linear(samples, MODEL_RATE, TARGET_RATE)
        pcm = samples.astype("<f4").tobytes()
        self._respond(200, pcm, "application/octet-stream")

    def _respond(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TtsHandler)
    print(f"TTS service listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
