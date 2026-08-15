# Голосовой ассистент (MVP) на Rust

Локальный голосовой AI-ассистент без облачных API. Пайплайн:

```
Микрофон → VAD (Silero) → STT (GigaAM) → LLM (Qwen) → TTS (Silero) → аудиовыход
```

## Стек

| Компонент | Технология | Модель |
|-----------|-----------|--------|
| Audio I/O (Linux) | CPAL 0.18 | — |
| Audio I/O (Termux) | Android AAudio (нативный) | — |
| VAD | Silero VAD (ONNX Runtime, `ort`) | `silero_vad.onnx` |
| STT | transcribe.cpp (Rust bindings) | GigaAM v3 e2e-CTC Q8_0 |
| LLM | llama.cpp (`llama-cpp-2`) | Qwen2.5-0.5B-Instruct q4_k_m (Termux) / Qwen3-0.6B (Linux) |
| TTS | Python-сервис (stdlib HTTP + Silero) | `v5_5_ru` |

Внутренний формат аудио — mono / 16 kHz / f32.

## Архитектура

```
src/
├── audio/      CPAL захват + воспроизведение, ресемплинг (Linux)
├── aaudio/     нативный AAudio: захват + воспроизведение (Termux)
├── vad/        Silero VAD (орт-обёртка + стейт-машина)
├── stt/        transcribe.cpp (partial transcripts)
├── llm/        llama.cpp (потоковая генерация)
├── tts/        HTTP-клиент к Python-сервису
├── pipeline/   оркестрация на Tokio-каналах + barge-in (Linux)
├── stream/     потоковый цикл VAD→STT→LLM→TTS на AAudio (Termux)
├── offline/    файловый режим in.wav→...→out.wav (Termux, запасной)
├── resample/   линейный ресемплер mono f32
├── text/       нормализация цифр + фильтр think-блоков
├── traits.rs   контракты компонентов (граница заменяемости)
├── types.rs    общие типы (AudioChunk, события)
└── config.rs   TOML-конфигурация
```

Каждый ML-бэкенд скрыт за trait'ом (`Vad`, `SpeechToText`, `Llm`,
`TextToSpeech`, `AudioCapture`, `AudioSink`), поэтому отдельные компоненты
можно менять без переписывания остального (ТЗ §9).

Все компоненты работают параллельно через async-каналы Tokio. LLM генерирует
ответ потоково; текст режется на предложения и отдаётся в TTS, не дожидаясь
конца генерации. При начале новой речи пользователя (barge-in) текущий
ответ отменяется, а буфер воспроизведения очищается.

## Сборка

Системные зависимости (Debian/Ubuntu):

```bash
sudo apt install build-essential cmake pkg-config clang libasound2-dev
```

```bash
cargo build --release
```

Первый запуск сборки долгий: `ort` качает ONNX Runtime, `transcribe-cpp` и
`llama-cpp-2` компилируют нативный C++ (transcribe.cpp / llama.cpp) через CMake.

## Модели

```bash
./download_models.sh all      # или vad | stt | llm
```

Скачивает в `models/`:
- `silero_vad.onnx` (~2 МБ)
- `gigaam-v3-e2e-ctc-Q8_0.gguf` (~272 МБ)
- `Qwen3-0.6B-abliterated-q4_k_m.gguf` (~397 МБ) — для Linux

Для Termux используется быстрая non-reasoning модель **Qwen2.5-0.5B-Instruct** —
её надо скачать отдельно (отвечает ~в 4 раза быстрее Qwen3, без блока
размышлений). Путь к ней задаётся в `config.termux.toml`:

```bash
curl -L --fail -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

## Termux (Android)

Ассистент работает на телефоне в Termux. Так как CPAL на Android не может
открыть микрофон/колонки, используется **нативный аудио через Android AAudio**,
встроенный прямо в бинарь. Это даёт **потоковый режим**: микрофон непрерывно
пишет PCM в бинарь, VAD по чанкам 30 мс сам определяет начало и конец речи
(авто-стоп по тишине), и ответ озвучивается через AAudio же. Никаких временных
файлов и termux-microphone-record не нужно.

### Установка

```bash
# в Termux:
pkg update && pkg upgrade
pkg install onnxruntime curl \
    python python-torch python-numpy
```

> **torch ставится через `pkg`, не через pip.** В Termux `python-torch`
> (CPU-only, ~aarch64) уже включает `python-numpy`. Никакого venv не нужно.

**Бинарь собирается на десктопе кросс-компиляцией** (см. ниже) и копируется в
Termux. Внутри Termux его не собрать — `llama-cpp-2` требует Android NDK,
которого в Termux нет.

### Запуск (потоковый режим, рекомендуется)

```bash
# 1) TTS-сервис (Python) в отдельном терминале:
python tts_service/service.py --host 127.0.0.1 --port 8090

# 2) потоковый цикл (говори — телефон отвечает):
LD_LIBRARY_PATH="$(pwd)" ./voice-assistant-android --stream \
    --config config.termux.toml
```

`--stream` можно опустить — на Termux это режим по умолчанию.

### Запуск (файловый режим, запасной)

Одноразовая реплика через файлы (если AAudio недоступен):

```bash
./termux_voice.sh once     # один раз
./termux_voice.sh loop     # бесконечно (термин-запись → ответ → воспроизведение)
```

`termux_voice.sh` использует `termux-microphone-record` (только m4a/amr/opus —
не WAV), декодирует через ffmpeg в mono 16k, прогоняет пайплайн и играет ответ
через `termux-media-player`.

Замечания для Termux:

- `config.termux.toml` — конфиг для телефона: `vad.threshold = 0.5`,
  `llm.max_tokens = 64`, модель `qwen2.5-0.5b-instruct-q4_k_m.gguf`. В нём нет
  костыля `input_sample_rate` (он есть только в `config.toml` под
  Logitech-вебкамеру на Linux).
- Модели и бинарь должны лежать на устройстве (см. раздел «Модели» и
  «Кросс-компиляция» ниже).
- TTS-сервис работает на CPU — на слабом телефоне синтез может занимать
  заметное время (это ожидаемо).

### LLM (llama.cpp) на Termux: кросс-компиляция

LLM-бэкенд через крейт `llama-cpp-2` **нельзя собрать внутри Termux**: его
build.rs на `aarch64-linux-android` требует Android NDK (которого в Termux
нет), в любой версии (в т.ч. 0.1.154). Поэтому под Android бинарь собирается
**кросс-компиляцией на десктопе** с Android NDK, а готовый ELF кладётся в
Termux. Требование NDK при этом удовлетворяется (NDK есть на хосте).

Самый простой путь — скрипт `build_termux_cross.sh` (сам находит NDK, ставит
rust-target при необходимости, делает стабы `libpthread.so`/`libaaudio.so` для
линковки и кладёт NDK-версию `libc++_shared.so` рядом с бинарём):

```bash
# на десктопе (Linux):
rustup target add aarch64-linux-android
cargo install cargo-ndk
# скачать Android NDK (напр. r27c) в ~/Android/Sdk/ и указать:
export ANDROID_NDK_ROOT=~/Android/Sdk/android-ndk-r27c
./build_termux_cross.sh
# результат: target/aarch64-linux-android/release/voice-assistant
#   + libc++_shared.so рядом
```

Вручную через `cargo ndk`:

```bash
export ANDROID_NDK_ROOT=~/Android/Sdk/android-ndk-r27c
# -P 24 (Android 7.0): bionic exposes POSIX_MADV_* only when __ANDROID_API__ >= 23
cargo ndk -t arm64-v8a -P 24 build --release
```

Замечания:

- **AAudio** (`libaaudio.so`) — системная Android-либа, которой нет в NDK
  aarch64 sysroot. Для кросс-линковки `build_termux_cross.sh` генерирует стаб с
  нужными символами; в рантайме загружается настоящая системная либа.
- **`libpthread.so`** — в bionic pthread встроен в libc, отдельной либы нет.
  `transcribe-cpp` линкует `-lpthread`, поэтому скрипт кладёт пустой стаб.
- `ort` на android использует `load-dynamic` (см. Cargo.toml): бинарь в
  рантайме dlopen'ит `libonnxruntime.so`. На устройстве он есть в
  `$PREFIX/lib` (пакет `onnxruntime`).
- Бинарь линкуется с `libc++_shared.so`; Termux ставит более старый libc++, чей
  ABI не содержит некоторых символов NDK-r27. Поэтому рядом с бинарём кладётся
  NDK-версия `libc++_shared.so`, и запускать нужно с
  `LD_LIBRARY_PATH=<каталог с бинарём>`.
- `llama-cpp-sys-2` и `transcribe-cpp` собираются нативно через NDK-toolchain.

Работоспособность потокового пайплайна на устройстве проверена: AAudio-захват,
VAD→STT→LLM→TTS и AAudio-воспроизведение отрабатывают в непрерывном цикле
(см. раздел «Запуск (потоковый режим)»).

## TTS-сервис (Python)

Сервис использует только stdlib (`http.server.ThreadingHTTPServer`), поэтому
web-фреймворк не нужен. На Linux зависимости (torch, numpy) ставятся через
pip:

```bash
cd tts_service
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python service.py --host 127.0.0.1 --port 8090
```

В Termux torch и numpy ставятся через `pkg install python-torch python-numpy`,
и сервис запускается системным python без venv (см. раздел Termux выше).

Первый запуск скачивает Silero TTS `v5_5_ru` с torch.hub.

## Запуск

1. Запустить TTS-сервис (см. выше).
2. Запустить ассистента:

```bash
cargo run --release -- --config config.toml
```

Говорите в микрофон → речь распознаётся → ответ начинает озвучиваться по мере
генерации. Ctrl-C для выхода.

## Конфигурация (`config.toml`)

- `audio` — устройство ввода/вывода, sample_rate, размер чанка,
  а также `input_sample_rate` / `output_sample_rate` (опциональное
  переопределение частоты устройства — см. ниже).
- `vad` — путь модели, `threshold`, `min_silence_ms`.
- `stt` — путь модели, язык, интервал partial-распознавания.
- `llm` — путь модели, `n_ctx`, `max_tokens`, `temperature`, `system_prompt`.
- `tts` — URL Python-сервиса, `speaker`, таймаут.

## Отладка и известные грабли

### Речь звучит ускоренно («чипманк»), вход распознаётся бредом

Частая причина — рассинхрон частоты: ALSA/pulseaudio докладывает одну
частоту, а устройство реально работает на другой. В этом проекте так было с
USB-вебкамерой Logitech `0x46d:0x821`, которая нативно работает на
**32000 Гц**, а CPAL сообщал 48000 Гц. Из-за этого ресемплер делил на 3
вместо 2, и входной звук сжимался в 1.5 раза (STT выдавал обрывки/бред).

Диагностика фактической частоты:

```bash
pactl list sources | grep -A2 "Name:"          # Sample Specification
cat /proc/asound/card*/pcm*/sub*/hw_params      # rate устройства
```

Исправление — переопределить частоту в `config.toml`:

```toml
[audio]
input_sample_rate = 32000   # реальная частота микрофона
# output_sample_rate = 48000 # аналогично для вывода, если нужно
```

Проверить, что override применился, — в логе появится строка
`overriding input sample rate reported=... target=...`.

### Стерео-вывод: не вытаскивать сэмпл на каждый канал

Очередь воспроизведения хранит **моно**-сэмплы. В output-колбэке нужно
вытаскивать **один** сэмпл на фрейм и писать его во все каналы:

```rust
for _ in 0..frames {
    let s = q.pop_front().unwrap_or(0.0f32); // ОДИН сэмпл на фрейм
    for _ in 0..channels {
        data[idx] = T::from_sample(s);        // тот же сэмпл во все каналы
        idx += 1;
    }
}
```

Если по ошибке вызывать `pop_front()` внутри цикла по каналам, очередь
моно-аудио дренируется в `channels` раз быстрее (для стерео — 2× ускорение
+ левый/правый канал получают чётные/нечётные сэмплы).

### Несколько процессов ассистента одновременно

Если слышно несколько голосов параллельно (один ускоренный, один
нормальный) — значит, запущено несколько экземпляров. Убивать нужно сам
бинарь, а не bash-обёртку (обёртка умирает, а дочерний процесс осиротевает
с parent=1). Проверка и очистка:

```bash
ps -ef | grep "voice-assistant --config" | grep -v grep   # сколько их
kill -9 <pid>                                              # каждый
ps -ef | grep "voice-assistant --config" | grep -v grep | wc -l   # должно быть 1
```

Запускать через `exec` (`exec ./target/debug/voice-assistant ...`), чтобы PID
указывал прямо на бинарь, а не на обёртку.

### Логирование цепочки

На Linux логи идут в stdout/stderr (tracing). На Termux потоковый режим
направляется в файл, напр. `~/stream.log`. Ключевые события на цикл:

```bash
# Linux:
tail -f /tmp/va_live.log | grep -aE "final transcript|llm generated|synthesizing"
# Termux (потоковый):
tail -f ~/stream.log | grep -aE "final transcript|llm generated|synthesized reply"
```

- `final transcript` — что распознал STT (услышано);
- `llm generated` — полный ответ LLM (включая отфильтрованные think-блоки);
- `synthesizing tts_chunk` — текст, фактически ушедший в TTS.

`grep` требует `-a`, т.к. в лог попадают не-UTF8 байты из вывода STT/LLM.

### Начальный согласный «р» отрезается (VAD)

Silero VAD может не уловить тихий слово-начальный дрожащий «р» — его энергия
на границе речи ниже порога, и VAD «просыпается» только на следующем
гласном, отрезая «р». Фикс — пре-ролл: в `vad_stt_driver` перед началом речи
в буфер STT добавляется один предыдущий чанк (~30 мс), захватывающий начальный
согласный.

### Цифры не озвучиваются (TTS)

Silero TTS `v5_5_ru` фильтрует вход по своему символьному алфавиту и молча
выбрасывает всё, чего в нём нет — включая ASCII-цифры 0-9. Фикс —
`src/text.rs::normalize_digits()`: перед отправкой в TTS цифровые
последовательности переводятся в русские слова («123» → «сто двадцать три»,
«2024» → «две тысячи двадцать четыре»). Реализован полный num2words до
миллиардов с учётом форм «одна/две тысячи» и склонений «тысяча/тысячи/тысяч».

### Шумные логи загрузки модели (llama.cpp)

llama.cpp/GGML печатают логи загрузки модели («sched_reserve»,
«graph_reserve», «Flash Attention», «Gated Delta Net») в stderr напрямую,
минуя `tracing` — поэтому `RUST_LOG` их не фильтрует. Глушатся вызовом
`send_logs_to_tracing(LogOptions::default().with_logs_enabled(false))` в
`QwenLlm::new()`. Реальные ошибки при этом не теряются — они возвращаются
через `Result` и логируются кодом.

## Замечания по зависимостям

Версии нативных бэкендов зафиксированы и должны обновляться осознанно
(см. `DEPENDENCIES.md`):
- `ort = "=2.0.0-rc.10"` — из актуального rust-example Silero VAD.
- `transcribe-cpp` — закреплён commit `856d7c1`, `default-features = false`
  (без macOS-`metal`).
- `llama-cpp-2 = "=0.1.153"` + `llama-cpp-sys-2 = "=0.1.153"` — пара
  закреплена, т.к. `-sys` 0.1.154 меняет FFI-раскладку и ломает обёртку.
