# Актуальная документация и исходники

Проверено: 15 августа 2026 года.

## Rust / runtime

### Tokio
https://docs.rs/tokio/latest/tokio/

Асинхронный runtime и каналы между компонентами.

### CPAL
https://docs.rs/cpal/latest/cpal/

Низкоуровневый Rust API для audio input/output. Текущая документация указывает версию 0.18.1.

### Serde
https://docs.rs/serde/latest/serde/

Сериализация конфигурации и структур данных.

### TOML
https://docs.rs/toml/latest/toml/

Чтение конфигурации приложения.

### Reqwest
https://docs.rs/reqwest/latest/reqwest/

HTTP-клиент Rust → Python/Robyn TTS service.

### Tracing
https://docs.rs/tracing/latest/tracing/

Структурированное логирование и диагностика.

---

## VAD

### Silero VAD
https://github.com/snakers4/silero-vad

Основной репозиторий Silero VAD.

### Silero VAD — Rust example
https://github.com/snakers4/silero-vad/tree/master/examples/rust-example

Использовать этот example как источник актуальных зависимостей и способа интеграции ONNX Runtime.

Важно: версию `ort` не угадывать. Перед подключением сверить актуальный `Cargo.toml` этого example и зафиксировать именно совместимую версию.

### Silero VAD — документация по примерам и зависимостям
https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies

---

## STT

### transcribe.cpp
https://github.com/handy-computer/transcribe.cpp

Актуальный C/C++ backend для GGUF STT-моделей.

Поддерживает GigaAM и другие семейства моделей.

### Rust bindings
https://github.com/handy-computer/transcribe.cpp/tree/main/bindings/rust/transcribe-cpp

Использовать официальный Rust binding.

Для стабильной сборки желательно pin'ить конкретный Git commit, а не использовать `main`.

### Документация GigaAM
https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/gigaam.md

### Модель GigaAM v3 e2e-CTC
https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf

MVP-модель:

https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q8_0.gguf

transcribe.cpp указывает для STT входной формат 16 kHz mono WAV.

---

## LLM

### llama-cpp-2
https://docs.rs/llama-cpp-2/latest/llama_cpp_2/

Текущая документация показывает версию 0.1.153.

Важно: crate представляет близкие к llama.cpp bindings, а API llama.cpp быстро меняется. Поэтому для воспроизводимой сборки использовать exact version и Cargo.lock.

### llama-cpp-rs исходный репозиторий
https://github.com/utilityai/llama-cpp-rs

### Модель Qwen 0.6B
https://huggingface.co/Mungert/Qwen3-0.6B-abliterated-GGUF

MVP-модель:

https://huggingface.co/Mungert/Qwen3-0.6B-abliterated-GGUF/blob/main/Qwen3-0.6B-abliterated-q4_k_m.gguf

---

## TTS

### Silero Models
https://github.com/snakers4/silero-models

Используем русскую модель `v5_5_ru`.

### Robyn
https://github.com/sparckles/robyn

Python web framework для отдельного локального TTS HTTP-сервиса.

---

## Системная сборка Linux

Для Ubuntu/Debian базовый набор:

```bash
sudo apt update
sudo apt install     build-essential     cmake     pkg-config     clang     libasound2-dev
```

Дальнейшие native-зависимости проверять по актуальным build-инструкциям конкретных проектов.

---

## Правила фиксации зависимостей

1. Хранить `Cargo.lock` в репозитории.
2. Для `llama-cpp-2` использовать exact version.
3. Для `transcribe.cpp` желательно pin конкретного Git commit.
4. Для `ort` брать версию из актуального Silero Rust example.
5. Не делать массовый `cargo update` без необходимости.
6. После успешной сборки сохранить рабочий набор версий вместе с lockfile.

---

## Основные ссылки

Rust audio:
https://docs.rs/cpal/latest/cpal/

Async:
https://docs.rs/tokio/latest/tokio/

VAD:
https://github.com/snakers4/silero-vad

VAD Rust example:
https://github.com/snakers4/silero-vad/tree/master/examples/rust-example

STT:
https://github.com/handy-computer/transcribe.cpp

STT Rust:
https://github.com/handy-computer/transcribe.cpp/tree/main/bindings/rust/transcribe-cpp

GigaAM:
https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/gigaam.md

LLM:
https://docs.rs/llama-cpp-2/latest/llama_cpp_2/

TTS:
https://github.com/snakers4/silero-models

Robyn:
https://github.com/sparckles/robyn
