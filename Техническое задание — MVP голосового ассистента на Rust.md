# Техническое задание — MVP голосового ассистента на Rust

## 1. Цель

Сделать локальный голосовой AI-ассистент с основным backend на Rust.

Целевая ОС:

- Linux
- в будущем возможен запуск в Termux на Android

Главная задача MVP:

```text
Микрофон
→ VAD
→ STT
→ LLM
→ TTS
→ аудиовыход
```

Ассистент должен работать без облачных API.

---

## 2. Используемый стек

### VAD

Использовать Silero VAD.

Rust example:

https://github.com/snakers4/silero-vad/tree/master/examples/rust-example

Задача VAD:

- определять начало речи;
- определять конец речи;
- передавать аудио STT уже только в рамках речевой реплики.

Важно: STT должен начинать работать до того, как пользователь закончил говорить.

---

### STT

Использовать transcribe.cpp с Rust bindings:

https://github.com/handy-computer/transcribe.cpp/tree/main/bindings/rust/transcribe-cpp

Модель:

https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q8_0.gguf

STT должен поддерживать partial transcript во время речи пользователя.

Поскольку выбранная модель не является полноценной streaming-моделью, допускается повторный запуск распознавания по мере накопления аудио.

Важно сделать STT отдельным модулем, чтобы позже можно было заменить модель.

---

### LLM

Использовать llama.cpp через Rust bindings `llama-cpp-2`.

Модель для MVP:

https://huggingface.co/Mungert/Qwen3-0.6B-abliterated-GGUF/blob/main/Qwen3-0.6B-abliterated-q4_k_m.gguf

LLM должна генерировать ответ потоково, то есть выдавать токены по мере генерации.

Не нужно ждать окончания полной генерации ответа.

---

### TTS

Использовать Silero TTS `v5_5_ru`.

https://github.com/snakers4/silero-models

Поскольку для Silero TTS нет нужного Rust binding, использовать отдельный Python-сервис на Robyn:

https://github.com/sparckles/robyn

Rust-приложение отправляет текст в Python через HTTP.

Python генерирует аудио и возвращает его Rust-приложению.

На первом этапе допускается обычный HTTP request/response без сложного streaming API.

---

## 3. Основной pipeline

Основной pipeline должен выглядеть так:

```text
Microphone
   ↓
Audio capture
   ↓
Silero VAD
   ↓
STT
   ↓
Final transcript
   ↓
LLM
   ↓
Text chunks
   ↓
Silero TTS
   ↓
Audio playback
```

Но LLM и TTS должны работать параллельно.

Например:

```text
LLM генерирует:
"Ближайшая заправка находится..."

          ↓

TTS начинает озвучивать эту часть

          ↓

LLM продолжает:
"...в двух километрах от вас."
```

Таким образом, не нужно ждать полного ответа LLM перед запуском TTS.

---

## 4. Audio

Внутренний формат аудио:

```text
mono
16000 Hz
f32
```

Микрофон должен быть преобразован к этому формату до передачи в VAD/STT.

Аудиозахват и аудиовывод должны находиться в Rust-приложении.

Python-сервис нужен только для TTS.

---

## 5. Архитектура Rust проекта

Необходимо разделить проект на независимые модули:

```text
audio
vad
stt
llm
tts
pipeline
```

Пример:

```text
src/
├── audio/
├── vad/
├── stt/
├── llm/
├── tts/
└── pipeline/
```

Каждый ML backend должен находиться в своём модуле.

Не связывать VAD напрямую с конкретной реализацией STT, STT напрямую с llama.cpp и т.д.

---

## 6. Асинхронная работа

Компоненты не должны работать строго последовательно.

Во время работы ассистента одновременно могут выполняться:

```text
VAD
STT
LLM
TTS
audio playback
```

Для коммуникации между компонентами можно использовать async channels Tokio.

---

## 7. Barge-in

Ассистент должен уметь реагировать на пользователя, пока сам говорит.

Пример:

```text
Ассистент говорит
        ↓
Пользователь начинает говорить
        ↓
VAD обнаруживает речь
        ↓
Остановить текущий TTS / playback
        ↓
Начать обработку новой реплики
```

На первом MVP достаточно корректно остановить воспроизведение и начать новый запрос.

---

## 8. Что пока НЕ реализовывать

Не реализовывать на этом этапе:

```text
RAG
embeddings
memory
intent classifier
MCP
tools
агентов
GUI
облачные API
```

Нужен только рабочий voice pipeline.

---

## 9. Главное требование

Архитектура должна позволять позже заменить отдельные компоненты.

Например:

```text
GigaAM
   ↓
другая STT-модель
```

или:

```text
Qwen 0.6B
   ↓
другая GGUF-модель
```

без переписывания всего приложения.

---

## 10. Результат MVP

На выходе должно получиться приложение:

```text
Rust application
    │
    ├── microphone
    ├── Silero VAD
    ├── GigaAM STT
    ├── llama.cpp / Qwen
    ├── HTTP client
    │
    └── audio playback

Python service
    │
    └── Robyn + Silero TTS
```

Пользователь говорит в микрофон → речь распознаётся → запрос передаётся LLM → ответ начинает генерироваться → первый готовый фрагмент отправляется в TTS → ассистент начинает говорить, не дожидаясь полного ответа LLM.