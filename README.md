# Голосовой ассистент (Rust)

Локальный голосовой AI-ассистент без облака:

```
Микрофон → VAD (Silero) → STT (GigaAM) → LLM (Qwen) → TTS (Silero) → аудиовыход
```

![Схема голосового ассистента](docs/voice-assistant.png)

Работает на Linux (CPAL) и в Termux/Android (нативный AAudio, потоковый режим).
Внутренний формат аудио — mono / 16 kHz / f32.

---

## Запуск

### Linux

Системные зависимости:

```bash
sudo apt install build-essential cmake pkg-config clang libasound2-dev
```

Сборка и модели:

```bash
cargo build --release
./download_models.sh all
```

TTS-сервис (Python, нужен torch/numpy):

```bash
cd tts_service && python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python service.py --host 127.0.0.1 --port 8090
```

Запуск ассистента (в другом терминале):

```bash
cargo run --release -- --config config.toml
```

Говорите в микрофон — ответ озвучивается по мере генерации. Ctrl-C — выход.

### Termux (Android)

Бинарь **кросс-компилируется на десктопе** (внутри Termux его не собрать —
`llama-cpp-2` требует Android NDK, которого там нет):

```bash
# на десктопе — системные зависимости для сборки с NDK:
sudo apt install build-essential cmake pkg-config clang

# и инструменты Rust для android-таргета:
rustup target add aarch64-linux-android
cargo install cargo-ndk

# скачать Android NDK (напр. r27c) и распаковать, затем:
export ANDROID_NDK_ROOT=~/Android/Sdk/android-ndk-r27c
./build_termux_cross.sh    # → target/aarch64-linux-android/release/voice-assistant
# скопируйте бинарь + libc++_shared.so в Termux, напр. в ~/va/
```

В Termux:

```bash
pkg install onnxruntime curl python python-torch python-numpy
# скачать модели (VAD, STT, и быструю Qwen2.5-0.5B):
./download_models.sh all
curl -L --fail -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# TTS-сервис (torch/numpy уже стоят через pkg, venv не нужен):
python tts_service/service.py --host 127.0.0.1 --port 8090

# потоковый режим (говори — телефон отвечает):
cd ~/va && LD_LIBRARY_PATH="$(pwd)" ./voice-assistant --config config.termux.toml
```

Замечания:
- На Termux бинарь dlopen'ит системный `libonnxruntime.so` (пакет `onnxruntime`) и
  требует NDK-версию `libc++_shared.so` рядом — потому и `LD_LIBRARY_PATH`.
- `config.termux.toml` настроен под телефон: порог VAD 0.5, `max_tokens = 64`,
  модель `qwen2.5-0.5b-instruct-q4_k_m.gguf` (быстрая, без блока размышлений).

---

## Модели

`./download_models.sh all` качает в `models/`:
- `silero_vad.onnx` (~2 МБ)
- `gigaam-v3-e2e-ctc-Q8_0.gguf` (~272 МБ)
- `Qwen3-0.6B-abliterated-q4_k_m.gguf` (~397 МБ) — для Linux

Для Termux дополнительно нужна `qwen2.5-0.5b-instruct-q4_k_m.gguf` (~491 МБ) —
см. команду выше.

---

## Структура

```
src/
├── audio/      CPAL захват + воспроизведение (Linux)
├── aaudio/     нативный AAudio: захват + воспроизведение (Termux)
├── vad/        Silero VAD (орт + стейт-машина)
├── stt/        transcribe.cpp (GigaAM)
├── llm/        llama.cpp (Qwen)
├── tts/        HTTP-клиент к Python-сервису
├── pipeline/   оркестрация на Tokio-каналах + barge-in (Linux)
├── stream/     потоковый цикл VAD→STT→LLM→TTS на AAudio (Termux)
└── text/       нормализация цифр + фильтр think-блоков
```

Каждый ML-бэкенд за trait'ом (`Vad`, `SpeechToText`, `Llm`, `TextToSpeech`,
`AudioCapture`, `AudioSink`) — компоненты меняются без переписывания остального.

Замечания по закреплённым версиям нативных бэкендов — см. `DEPENDENCIES.md`.
