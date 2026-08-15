import json
import urllib.request

import numpy as np
import wave

text = "Привет! Это тест голосового ассистента. Раз, два, три, четыре, пять."
data = json.dumps({"text": text, "speaker": "xenia"}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8090/tts", data=data, headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req, timeout=60)
pcm = resp.read()

samples = np.frombuffer(pcm, dtype="<f4")
peak = float(np.abs(samples).max()) if len(samples) else 0.0
scale = 32767.0 / max(peak, 1e-6)
samples16 = (samples * scale).astype("<i2")

with wave.open("/tmp/tts_test.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(samples16.tobytes())

print(f"samples={len(samples)}  duration={len(samples)/16000:.2f}s  peak={peak:.3f}")
print("wrote /tmp/tts_test.wav (16 kHz mono s16le)")
