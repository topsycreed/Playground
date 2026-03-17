# Telegram MP3 -> Text Bot (RU/EN, free, local)

This bot transcribes speech from Telegram audio files to text using `faster-whisper` (local inference, no paid API).

Features:
- Russian (`ru`), English (`en`), Auto language detection (`auto`)
- Interactive Telegram settings menu (`/help` or `/settings`)
- Quality profiles (`fast`, `balanced`, `best`)
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
- Python 3.10+ (project is running on Python 3.14 in this workspace)
- Internet access on first model download
- Telegram bot token from `@BotFather`

For CUDA mode only:
- NVIDIA driver (latest stable)
- CUDA Toolkit 12.x
- cuDNN 9.x for CUDA 12

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
```

## 6) Run Bot

```powershell
python bot.py
```

On first run, model files are downloaded from Hugging Face.
They are cached locally, so re-download is not needed every start (unless cache is removed or model changes).

## 7) Use Bot in Telegram

- Send MP3/audio/voice file
- `/help` or `/settings` opens interactive menu
- Menu language options: `auto`, `ru`, `en`
- Menu quality options: `fast`, `balanced`, `best`
- Text commands also work:
`/lang auto|ru|en`
`/quality fast|balanced|best`
`/gpu` (show current model/device/compute runtime status)
- `/gpu` also reports if `cublas64_12.dll` and `cudnn64_9.dll` are visible in PATH

## Troubleshooting

`RuntimeError: cublas64_12.dll is not found`
- Install CUDA Toolkit 12.x
- Ensure CUDA `bin` path is in `PATH`
- Open a new terminal

`RuntimeError: cudnn64_9.dll is not found`
- Install cuDNN 9.x for CUDA 12
- Ensure cuDNN DLL location is in `PATH` (or copied into CUDA `bin`)

Bot starts but transcriptions are inaccurate
- Use `WHISPER_MODEL_SIZE=large-v3`
- Use menu quality `best`
- Set language explicitly (`ru` or `en`) if audio is mono-language
- Keep `WHISPER_VAD_FILTER=false` if words are getting dropped

Very slow transcription
- Enable CUDA (`WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`)
- Use menu quality `balanced` or `fast`
- Reduce model size (`medium`, `small`) if needed

## Security Note

If a bot token was ever exposed in logs/chat, revoke and regenerate it in `@BotFather`, then update `.env`.
