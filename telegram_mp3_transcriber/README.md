# Telegram MP3 to Text Bot (RU/EN, free)

This bot transcribes audio to text using a **free local** model (`faster-whisper`).
It supports:

- Russian (`ru`)
- English (`en`)
- Auto-detect (`auto`)

If audio is too large, the bot automatically:

1. Splits it into smaller chunks
2. Transcribes each chunk
3. Concatenates the text into one result

## 1) Create Telegram bot token

1. Open Telegram and chat with `@BotFather`
2. Run `/newbot`
3. Copy your bot token

## 2) Install dependencies

From this folder (`telegram_mp3_transcriber`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

- `TELEGRAM_BOT_TOKEN`

Optional tuning:

- `WHISPER_MODEL_SIZE` (`large-v3` for best quality, `small` for speed)
- `WHISPER_DEVICE` (`cuda` for NVIDIA GPU, `cpu` fallback)
- `WHISPER_COMPUTE_TYPE` (`float16` on GPU, `int8` on CPU)
- `TARGET_CHUNK_MB` (`20` default)
- `MAX_CHUNK_SECONDS` (`900` default)
- `CHUNK_OVERLAP_SECONDS` (`2.0` recommended to avoid cutting words between chunks)
- `WHISPER_BEAM_SIZE` / `WHISPER_BEST_OF` (`8` for higher quality)
- `WHISPER_VAD_FILTER` (`false` recommended if words are missing)

## 4) Run bot

```powershell
python bot.py
```

On first start, whisper model files are downloaded once (internet required for this one-time download).

## Usage in Telegram

- Send voice/audio/MP3 file to bot
- `/help` or `/settings` opens interactive menu (buttons)
- Language switch in menu: `auto`, `ru`, `en`
- Quality switch in menu: `fast`, `balanced`, `best`
- `/lang auto|ru|en` also works
- `/quality fast|balanced|best` also works

## Notes

- 100% free to run (local inference, no paid API).
- No external `ffmpeg.exe` required (audio decoding is handled by `faster-whisper` + PyAV).
- If CUDA libs are missing, bot automatically falls back to CPU mode.
