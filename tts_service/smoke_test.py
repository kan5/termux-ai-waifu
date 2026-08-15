#!/usr/bin/env python3
"""Smoke-test the TTS HTTP service without downloading the real model.

Monkey-patches service.MODEL with a stub that returns a synthetic tensor,
starts the real ThreadingHTTPServer, POSTs a request, and verifies the
response is raw f32 LE PCM (mono 16 kHz).
"""
import json
import threading
import urllib.error
import urllib.request
import sys

import numpy as np
import torch  # for tensor construction only

# Import the service module (loads MODEL from torch.hub on import — but we
# replace it immediately after; to avoid a network download we patch the
# torch.hub.load used inside BEFORE importing).
import torch.hub
orig_hub_load = torch.hub.load


class _HubModel:
    """Stub with the surface load_model() touches (`.to`)."""

    def to(self, device):
        return self


def fake_hub_load(*args, **kwargs):
    return _HubModel(), None


torch.hub.load = fake_hub_load

import service  # noqa: E402

torch.hub.load = orig_hub_load


class StubModel:
    def apply_tts(self, text, speaker, sample_rate):
        # ~0.25 s of audio at requested rate.
        n = int(sample_rate * 0.25)
        t = np.linspace(0, 2 * np.pi * 8, n, dtype=np.float32)
        sig = (0.5 * np.sin(t)).astype(np.float32)
        return torch.from_numpy(sig)


service.MODEL = StubModel()

HOST, PORT = "127.0.0.1", 8091
server = service.ThreadingHTTPServer((HOST, PORT), service.TtsHandler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()

body = json.dumps({"text": "привет мир 2024", "speaker": "xenia"}).encode()
req = urllib.request.Request(
    f"http://{HOST}:{PORT}/tts",
    data=body,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req, timeout=30)
pcm = resp.read()
content_type = resp.headers.get("Content-Type")

samples = np.frombuffer(pcm, dtype="<f4")
duration = len(samples) / 16000.0

ok = content_type == "application/octet-stream" and len(samples) > 0
print(f"content_type = {content_type}")
print(f"bytes        = {len(pcm)}")
print(f"samples      = {len(samples)}  duration={duration:.3f}s")
print(f"peak         = {float(np.abs(samples).max()):.3f}")
print(f"mono@16k f32 = {pcm[:4].hex()}  (little-endian f32)")
print("SMOKE TEST:", "PASS" if ok else "FAIL")

# Empty text -> 400
try:
    req2 = urllib.request.Request(
        f"http://{HOST}:{PORT}/tts",
        data=json.dumps({"text": "  "}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req2, timeout=30)
    print("EMPTY TEST: FAIL (expected 400)")
except urllib.error.HTTPError as e:
    print("EMPTY TEST: PASS (got", e.code, ")")

server.shutdown()
server.server_close()
sys.exit(0 if ok else 1)
