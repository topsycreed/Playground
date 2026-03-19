# Telegram Audio/Video -> Text Bot (RU/EN, free, local)

## Quick Start

```powershell
cd .\telegram_mp3_transcriber\
.\.venv312\Scripts\Activate.ps1
python bot.py
```

This bot transcribes speech from Telegram audio/video files to text using `faster-whisper` (local inference, no paid API).

Features:
- Russian (`ru`), English (`en`), Auto language detection (`auto`)
- Interactive Telegram settings menu (`/help` or `/settings`)
- Quality profiles (`fast`, `balanced`, `best`)
- Output formats: plain text or dialogue (`User1` / `User2`)
- Input support: `audio`, `voice`, `video`, file attachments (`Document.AUDIO` / `Document.VIDEO`)
- MP4/video pipeline: audio track is extracted with `ffmpeg` and then transcribed
- Hybrid mode: Whisper transcription + optional NeMo diarization backend
- Optional LLM post-processing:
  - Google Gemini (AI Studio free tier)
  - Whisper (no LLM cleanup, raw transcript)
  - OpenAI oos-20 via LM Studio
- Full cycle output for audio/video: status message + `transcript.txt` file + summary (when post-processing is enabled)
- Debug text-file mode: cleanup + summary for `.txt/.md/.log/.json` documents
- Large-file handling: split into chunks + automatic text merge
- GPU (CUDA) acceleration with automatic CPU fallback
- Lazy model loading: bot starts immediately; model downloads/loads on first transcription

## Hardware Requirements

Minimum (CPU mode):
- 4+ CPU cores
- 8 GB RAM (16 GB recommended)
- 5+ GB free disk space

Recommended (GPU mode, better speed/quality):
- NVIDIA GPU with recent drivers (RTX series recommended)
- 8+ GB VRAM for `large-v3` comfort
- 16+ GB system RAM

## Software Requirements

- Windows 10/11 (PowerShell commands below target Windows)
- Python 3.10+ for Whisper bot
- Internet access on first model download
- Telegram bot token from `@BotFather`

For CUDA mode only:
- NVIDIA driver (latest stable)
- CUDA Toolkit 12.x
- cuDNN 9.x for CUDA 12

For NeMo diarization backend:
- `nemo_toolkit[asr]` dependencies (see `requirements-nemo.txt`)
- Recommended Python version: 3.11 or 3.12
- On Python 3.13+ (including 3.14), NeMo is often unavailable in this setup; bot will auto-fallback to heuristic diarization

## 1) Create Telegram Bot Token

1. Open Telegram and chat with `@BotFather`
2. Run `/newbot`
3. Copy the token
4. Save it for `.env`

## 2) Create Environment and Install Python Packages

From this folder:

```powershell
cd C:\Users\Gena\Documents\Playground\telegram_mp3_transcriber
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional for NeMo diarization:

```powershell
pip install -r requirements-nemo.txt
```

If NeMo install fails in your current environment (common on Python 3.13+), create a Python 3.12 venv:

```powershell
py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-nemo.txt
```

## 3) Configure `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` and set:

```env
TELEGRAM_BOT_TOKEN=PUT_YOUR_TOKEN_HERE
```

Optional if CUDA DLLs are not auto-detected:

```env
CUDA_EXTRA_PATHS=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64
```

NeMo diarization options:

```env
DIARIZATION_BACKEND=nemo
NEMO_NUM_SPEAKERS=0
NEMO_NUM_WORKERS=0
```

`NEMO_NUM_WORKERS=0` is recommended on Windows to avoid multiprocessing/pickling errors during NeMo diarization.

Gemini free-tier post-processing options:

```env
TEXT_POSTPROCESS_MODEL=gemini
# One key or multiple keys (comma/semicolon/newline separated):
GEMINI_API_KEYS=PUT_YOUR_KEY_1,PUT_YOUR_KEY_2
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_TIMEOUT_SEC=180
GEMINI_FALLBACK_MODEL=whisper
```

## 4) Enable CUDA (NVIDIA GPU) on Windows

If CUDA is not configured, bot still works on CPU, but slower.

1. Install or update NVIDIA driver
- Use NVIDIA app/GeForce Experience or official NVIDIA driver download.
- Reboot after installation.

2. Install CUDA Toolkit 12.x
- Install from NVIDIA CUDA Toolkit page.
- Keep default path like:
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

3. Install cuDNN 9.x for CUDA 12
- Download cuDNN for CUDA 12 from NVIDIA Developer.
- Extract and ensure `cudnn64_9.dll` is available in CUDA `bin` path or another path included in `PATH`.

4. Verify CUDA from terminal

```powershell
nvidia-smi
where.exe cublas64_12.dll
where.exe cudnn64_9.dll
```

If `where` does not find DLLs, add CUDA `bin` to system `PATH` (then open a new terminal):

- Typical CUDA bin path:
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

5. Verify Python can initialize GPU backend

```powershell
.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16'); print('CUDA OK')"
```

If this fails with `cublas64_12.dll` or `cudnn64_9.dll`, CUDA/cuDNN install or PATH is incomplete.

## 4.1) FFmpeg (required by NeMo/pydub path)

Also required for video input (`.mp4`, `.mov`, `.mkv`, etc.) because bot extracts audio track before transcription.

Install (if not installed):

```powershell
winget install Gyan.FFmpeg
```

Verify:

```powershell
where.exe ffmpeg
where.exe ffprobe
```

If not found, set in `.env`:

```env
FFMPEG_EXTRA_PATHS=C:\path\to\ffmpeg\bin
```

## 5) Recommended `.env` Profiles

Best quality on strong NVIDIA GPU:

```env
WHISPER_MODEL_SIZE=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
TARGET_CHUNK_MB=20
MAX_CHUNK_SECONDS=900
CHUNK_OVERLAP_SECONDS=2.0
WHISPER_BEAM_SIZE=8
WHISPER_BEST_OF=8
WHISPER_TEMPERATURE=0.0
WHISPER_VAD_FILTER=false
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

CPU-safe profile:

```env
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
CT2_ENABLE_TRANSFORMERS_CONVERTER=false
```

## 6) Run Bot

```powershell
python bot.py
```

On first run, model files are downloaded from Hugging Face.
They are cached locally, so re-download is not needed every start (unless cache is removed or model changes).

## 7) Use Bot in Telegram

- Send MP3/audio/voice or MP4 video
- Or send a text document (`.txt/.md/.log/.json/.csv/.yaml/.xml/.srt/.vtt`) for debug cleanup+summary
- For audio/video requests bot returns:
  - status message (`Done...`)
  - `transcript.txt` (full transcript as file)
  - summary (message or `summary.txt`, depending on length)
- `/help` or `/settings` opens interactive menu
- Menu language options: `auto`, `ru`, `en`
- Menu quality options: `fast`, `balanced`, `best`
- Menu format options: `text`, `dialog`
- Menu diarization options: `auto`, `nemo`, `heuristic`
- Menu speaker options: `auto`, `2`, `3`, `4`
- Menu post-process model options: `Google Gemini`, `Whisper`, `OpenAI oos-20`
- Text commands also work:
`/lang auto|ru|en`
`/quality fast|balanced|best`
`/format text|dialog`
`/model gemini|whisper|oos20`
`/diar auto|nemo|heuristic`
`/speakers auto|2|3|4`
`/gpu` (show current model/device/compute runtime status, including NeMo availability)
- `/llm` (show LM Studio post-processing status and selected model)
- `/gpu` also reports if `cublas64_12.dll` and `cudnn64_9.dll` are visible in PATH

## 8) Large Files (audio/video, 300 MB and more)

Telegram cloud Bot API limits bot downloads to 20 MB.  
For large files, run a local Telegram Bot API server.

According to official docs, local mode allows:
- Download files without a size limit
- Upload files up to 2000 MB

Sources:
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [tdlib/telegram-bot-api](https://github.com/tdlib/telegram-bot-api)

Bot `.env` settings for local server:

```env
TELEGRAM_LOCAL_MODE=true
TELEGRAM_API_BASE_URL=http://127.0.0.1:8081
TELEGRAM_API_FILE_URL=http://127.0.0.1:8081
```

One-command startup (`python bot.py`) with auto-launched local Bot API:

```env
TELEGRAM_LOCAL_MODE=true
TELEGRAM_AUTO_START_LOCAL_API=true
TELEGRAM_BOT_API_BIN=C:\path\to\telegram-bot-api.exe
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_LOCAL_API_HOST=127.0.0.1
TELEGRAM_LOCAL_API_PORT=8081
TELEGRAM_LOCAL_API_DATA_DIR=.telegram-bot-api-data
TELEGRAM_LOCAL_DOCKER_DATA_DIR=C:\Users\Gena\Documents\Playground\tg-bot-api-data
TELEGRAM_CONNECT_TIMEOUT=20
TELEGRAM_READ_TIMEOUT=120
TELEGRAM_WRITE_TIMEOUT=120
TELEGRAM_MEDIA_WRITE_TIMEOUT=180
TELEGRAM_POOL_TIMEOUT=15
TELEGRAM_GET_UPDATES_READ_TIMEOUT=30
TELEGRAM_LOCAL_DIRECT_PICKUP_MB=40
TELEGRAM_LOCAL_PICKUP_WAIT_SECONDS=30
```

Optional post-processing of final transcript (free, local LM Studio):
- Fix spelling/punctuation
- Remove filler words ("water")
- Replace `User1/User2` with names when speakers introduce themselves
- Generate concise summary

```env
TEXT_POSTPROCESS_ENABLED=true
TEXT_POSTPROCESS_PROVIDER=lmstudio
TEXT_POSTPROCESS_MODEL=gemini
GEMINI_API_KEYS=PUT_YOUR_KEY_1,PUT_YOUR_KEY_2
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_TIMEOUT_SEC=180
GEMINI_FALLBACK_MODEL=whisper
LMSTUDIO_BASE_URL=http://127.0.0.1:1234
LMSTUDIO_MODEL=
LMSTUDIO_PROMPTS_FILE=
LMSTUDIO_TIMEOUT_SEC=600
LMSTUDIO_TEMPERATURE=0.0
LMSTUDIO_REQUEST_RETRIES=2
LMSTUDIO_RETRY_BACKOFF_SEC=2.0
TEXT_POSTPROCESS_MAX_CHARS_PER_CHUNK=12000
TEXT_SUMMARY_MAX_CHARS_PER_CHUNK=20000
TEXT_POSTPROCESS_MIN_RESPLIT_CHARS=1200
TEXT_DEBUG_MAX_MB=50
```

Notes:
- Keep LM Studio running with a loaded chat model.
- `LMSTUDIO_BASE_URL` can be `http://127.0.0.1:1234`, `.../v1`, or `.../api/v1`.
- Bot auto-detects LM Studio API style (`/v1` or `/api/v1`).
- If `LMSTUDIO_MODEL` is empty, bot auto-picks the first model from `/models`.
- If LM Studio is unavailable, bot falls back to a lightweight heuristic cleanup.
- For debug text-file mode, bot decodes file content, runs cleanup, then sends separate summary + cleaned text.
- Speaker rename is guarded: if detected replacement is not likely a real name, bot keeps `User1/User2/...`.
- Prompt templates and normalization dictionary are stored in `lmstudio_prompts.json` (editable JSON).
- You can override location via `LMSTUDIO_PROMPTS_FILE`.
- For quality-first mode (slower): use lower `LMSTUDIO_TEMPERATURE`, smaller `TEXT_POSTPROCESS_MAX_CHARS_PER_CHUNK`, and higher `LMSTUDIO_TIMEOUT_SEC`.

Important:
- Before switching from cloud API to local API, call `logOut` once (per official docs).

Dialogue mode notes:
- Bot now auto-detects speaker count (heuristic local clustering).
- If one speaker is detected, output is plain text (no `UserX:` labels).
- If multiple speakers are detected, output is dialogue style (`User1`, `User2`, ...).
- It is heuristic, so labels can be imperfect for noisy/overlapping speech.
- When backend is `nemo` or `auto` and NeMo is installed, bot uses NeMo diarization first and falls back to heuristic if NeMo is unavailable.
- On Windows, bot prefers NeMo `ClusteringDiarizer` (more stable) and only uses `NeuralDiarizer` as secondary fallback.

## Troubleshooting

`RuntimeError: cublas64_12.dll is not found`
- Install CUDA Toolkit 12.x
- Ensure CUDA `bin` path is in `PATH`
- Open a new terminal

Bot hangs on startup around `import transformers` / `ctranslate2`
- Current build disables CTranslate2 `TransformersConverter` by default because it is not required for transcription.
- Keep `CT2_ENABLE_TRANSFORMERS_CONVERTER=false` in `.env` (recommended).
- If you explicitly need model conversion via ctranslate2 converters, set it to `true`.

`RuntimeError: cudnn64_9.dll is not found`
- Install cuDNN 9.x for CUDA 12
- Ensure cuDNN DLL location is in `PATH` (or copied into CUDA `bin`)

`BadRequest: File is too big`
- Telegram cloud Bot API can reject large file downloads (commonly over 20 MB)
- Bot now shows a friendly message instead of crashing
- Send smaller file, or split audio before sending to the bot
- You can configure the check via `.env`:
  `TELEGRAM_DOWNLOAD_LIMIT_MB=20`

`ffmpeg audio extraction failed` for MP4/video
- Ensure `ffmpeg.exe` is available in `PATH` (`where.exe ffmpeg`)
- If needed, set `FFMPEG_EXTRA_PATHS` in `.env`
- Retry with a video that has an actual audio track

Summary is missing after transcription
- Ensure `TEXT_POSTPROCESS_ENABLED=true`
- Check `/model` (for example `gemini`)
- Verify model endpoint with `/llm`

Bot starts but transcriptions are inaccurate
- Use `WHISPER_MODEL_SIZE=large-v3`
- Use menu quality `best`
- Set language explicitly (`ru` or `en`) if audio is mono-language
- Keep `WHISPER_VAD_FILTER=false` if words are getting dropped

Very slow transcription
- Enable CUDA (`WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`)
- Use menu quality `balanced` or `fast`
- Reduce model size (`medium`, `small`) if needed

`NeMo available: no` in `/gpu`
- Install NeMo requirements from `requirements-nemo.txt`
- Prefer Python 3.11/3.12 for NeMo environment
- Keep backend as `auto` if you want automatic fallback when NeMo is unavailable

`[WinError 4551] ... blocked this file` during NeMo diarization
- This is Windows Smart App Control / WDAC policy blocking external codec tools (commonly `ffprobe.exe`)
- Current bot build pre-converts audio to local PCM WAV before NeMo diarization to reduce this dependency
- If policy still blocks execution paths on your machine, use `/diar heuristic` as fallback

## Security Note

If a bot token was ever exposed in logs/chat, revoke and regenerate it in `@BotFather`, then update `.env`.
