import numpy as np
import torch

print("loading model...")
model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts", language="ru", speaker="v5_5_ru", trust_repo=True,
)
model.to("cpu")

text = "Привет, это тест."
a48 = model.apply_tts(text=text, speaker="xenia", sample_rate=48000)
a24 = model.apply_tts(text=text, speaker="xenia", sample_rate=24000)
a8 = model.apply_tts(text=text, speaker="xenia", sample_rate=8000)

a48 = a48.detach().cpu().numpy()
a24 = a24.detach().cpu().numpy()
a8 = a8.detach().cpu().numpy()

print("48k:", len(a48), "24k:", len(a24), "8k:", len(a8))

# If the model correctly resamples, then:
#   a24 == a48[::2] (decimate 48k by 2)
#   a8  == a48[::6] (decimate 48k by 6)
d2 = a48[::2]
d6 = a48[::6]

def err(a, b):
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).mean())

print("mean|a24 - a48[::2]| =", err(a24, d2))
print("mean|a8  - a48[::6]| =", err(a8, d6))
