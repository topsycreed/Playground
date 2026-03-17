from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from transcriber import SpeechTranscriber

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"auto", "ru", "en"}
SUPPORTED_QUALITIES = {"fast", "balanced", "best"}
user_language: dict[int, str] = {}
user_quality: dict[int, str] = {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_user_language(user_id: int) -> str:
    return user_language.get(user_id, "auto")


def _get_user_quality(user_id: int) -> str:
    return user_quality.get(user_id, "balanced")


def _find_dll_on_path(dll_name: str) -> str | None:
    for path_item in os.environ.get("PATH", "").split(os.pathsep):
        if not path_item:
            continue
        candidate = Path(path_item) / dll_name
        if candidate.exists():
            return str(candidate)
    return None


def _prepend_paths(paths: list[str]) -> None:
    current = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    current_lower = {p.lower() for p in current}
    to_add = []
    for path in paths:
        normalized = str(Path(path))
        if Path(normalized).exists() and normalized.lower() not in current_lower:
            to_add.append(normalized)
            current_lower.add(normalized.lower())
    if to_add:
        os.environ["PATH"] = os.pathsep.join(to_add + current)
        logger.info("Added %d CUDA/CUDNN path(s) to PATH for this bot process.", len(to_add))


def _configure_cuda_paths() -> None:
    env_extra = os.getenv("CUDA_EXTRA_PATHS", "").strip()
    user_paths = [p.strip() for p in env_extra.split(";") if p.strip()] if env_extra else []

    auto_paths: list[str] = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64",
    ]

    _prepend_paths(user_paths + auto_paths)


def _settings_text(user_id: int) -> str:
    lang = _get_user_language(user_id)
    quality = _get_user_quality(user_id)
    return (
        "Settings menu:\n"
        f"Language: {lang}\n"
        f"Quality: {quality}\n\n"
        "RU: Отправьте MP3/аудио/voice, и я сделаю транскрипт.\n"
        "EN: Send MP3/audio/voice and I will transcribe it."
    )


def _settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    lang = _get_user_language(user_id)
    quality = _get_user_quality(user_id)

    def mark(active: bool, label: str) -> str:
        return f"[x] {label}" if active else f"[ ] {label}"

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(mark(lang == "auto", "Lang Auto"), callback_data="lang:auto"),
                InlineKeyboardButton(mark(lang == "ru", "Lang RU"), callback_data="lang:ru"),
                InlineKeyboardButton(mark(lang == "en", "Lang EN"), callback_data="lang:en"),
            ],
            [
                InlineKeyboardButton(mark(quality == "fast", "Fast"), callback_data="quality:fast"),
                InlineKeyboardButton(
                    mark(quality == "balanced", "Balanced"),
                    callback_data="quality:balanced",
                ),
                InlineKeyboardButton(mark(quality == "best", "Best"), callback_data="quality:best"),
            ],
            [
                InlineKeyboardButton("Show current", callback_data="settings:show"),
                InlineKeyboardButton("Close menu", callback_data="settings:close"),
            ],
        ]
    )


async def _send_menu(message, user_id: int) -> None:
    await message.reply_text(
        _settings_text(user_id),
        reply_markup=_settings_keyboard(user_id),
    )


def _extract_file_info(update: Update) -> tuple[str, str]:
    message = update.effective_message
    if message is None:
        raise ValueError("No message to process.")

    if message.audio:
        ext = Path(message.audio.file_name or "audio.mp3").suffix or ".mp3"
        return message.audio.file_id, ext
    if message.voice:
        return message.voice.file_id, ".ogg"
    if message.document:
        ext = Path(message.document.file_name or "audio.bin").suffix or ".bin"
        return message.document.file_id, ext

    raise ValueError("No supported audio payload found.")


def _split_for_telegram(text: str, max_chars: int = 3900) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        candidate = paragraph if not current else f"{current}\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(paragraph) > max_chars:
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _format_gpu_status(transcriber: SpeechTranscriber) -> str:
    status = transcriber.status()
    model_loaded = "yes" if status["model_loaded"] else "no"
    loading = "yes" if status["loading"] else "no"
    state = "CUDA active" if status["active_device"] == "cuda" else "CPU mode active"
    cublas = _find_dll_on_path("cublas64_12.dll") or "not found"
    cudnn = _find_dll_on_path("cudnn64_9.dll") or "not found"
    return (
        "Runtime status:\n"
        f"Model: {status['model_size']}\n"
        f"Requested device: {status['requested_device']}\n"
        f"Active device: {status['active_device']}\n"
        f"Requested compute: {status['requested_compute_type']}\n"
        f"Active compute: {status['active_compute_type']}\n"
        f"Model loaded: {model_loaded}\n"
        f"Loading now: {loading}\n"
        f"cublas64_12.dll: {cublas}\n"
        f"cudnn64_9.dll: {cudnn}\n"
        f"State: {state}\n\n"
        "If active device is cpu while requested is cuda, CPU fallback is active."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    await message.reply_text(
        "Send me an MP3/audio file and I will convert speech to text.\n"
        "Use /help for interactive settings menu.\n"
        "Use /gpu to check CUDA/CPU runtime status.\n"
        "First transcription may take several minutes while model downloads/loads."
    )
    await _send_menu(message, user.id)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    await message.reply_text(
        "Commands:\n"
        "/help - open settings menu\n"
        "/settings - open settings menu\n"
        "/lang auto|ru|en\n"
        "/quality fast|balanced|best\n"
        "/gpu - show runtime device status\n\n"
        "Send audio as voice, audio, or file attachment."
    )
    await _send_menu(message, user.id)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    await _send_menu(message, user.id)


async def text_instructions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    await message.reply_text(
        "RU:\n"
        "Я распознаю речь из аудио файлов.\n"
        "Пожалуйста, отправьте MP3/аудио/voice-сообщение.\n\n"
        "EN:\n"
        "I transcribe speech from audio files.\n"
        "Please send an MP3/audio/voice message.\n\n"
        "Use /help for settings.\n"
        "Use /gpu to check runtime device."
    )
    await _send_menu(message, user.id)


async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    if not context.args:
        current = _get_user_language(user.id)
        await message.reply_text(f"Current language mode: {current}")
        return

    chosen = context.args[0].strip().lower()
    if chosen not in SUPPORTED_LANGUAGES:
        await message.reply_text("Use: /lang auto | /lang ru | /lang en")
        return

    user_language[user.id] = chosen
    await message.reply_text(f"Language mode set to: {chosen}")
    await _send_menu(message, user.id)


async def set_quality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    if not context.args:
        current = _get_user_quality(user.id)
        await message.reply_text(f"Current quality mode: {current}")
        return

    chosen = context.args[0].strip().lower()
    if chosen not in SUPPORTED_QUALITIES:
        await message.reply_text("Use: /quality fast | /quality balanced | /quality best")
        return

    user_quality[user.id] = chosen
    await message.reply_text(f"Quality mode set to: {chosen}")
    await _send_menu(message, user.id)


async def gpu_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    transcriber: SpeechTranscriber = context.application.bot_data.get("transcriber")
    if transcriber is None:
        await message.reply_text("Transcriber is not initialized yet.")
        return

    await message.reply_text(_format_gpu_status(transcriber))


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    query = update.callback_query
    user = update.effective_user
    if query is None or user is None:
        return

    data = query.data or ""
    if data.startswith("lang:"):
        chosen = data.split(":", 1)[1]
        if chosen in SUPPORTED_LANGUAGES:
            user_language[user.id] = chosen
            await query.answer(f"Language: {chosen}")
        else:
            await query.answer("Unsupported language", show_alert=True)
        await query.edit_message_text(
            _settings_text(user.id),
            reply_markup=_settings_keyboard(user.id),
        )
        return

    if data.startswith("quality:"):
        chosen = data.split(":", 1)[1]
        if chosen in SUPPORTED_QUALITIES:
            user_quality[user.id] = chosen
            await query.answer(f"Quality: {chosen}")
        else:
            await query.answer("Unsupported quality", show_alert=True)
        await query.edit_message_text(
            _settings_text(user.id),
            reply_markup=_settings_keyboard(user.id),
        )
        return

    if data == "settings:show":
        await query.answer("Updated")
        await query.edit_message_text(
            _settings_text(user.id),
            reply_markup=_settings_keyboard(user.id),
        )
        return

    if data == "settings:close":
        await query.answer("Closed")
        await query.edit_message_text(
            "Menu closed. Use /help or /settings to open it again."
        )
        return

    await query.answer("Unknown action")


async def transcribe_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    transcriber: SpeechTranscriber = context.application.bot_data["transcriber"]
    language = _get_user_language(user.id)
    quality = _get_user_quality(user.id)

    progress = await message.reply_text("Downloading audio...")
    try:
        file_id, file_ext = _extract_file_info(update)
        tg_file = await context.bot.get_file(file_id)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / f"input{file_ext}"
            await tg_file.download_to_drive(custom_path=str(input_path))

            if not transcriber.is_model_loaded():
                await progress.edit_text(
                    "Preparing model for first run (download + load). "
                    "Please wait, this can take a few minutes..."
                )
            else:
                await progress.edit_text(
                    f"Transcribing... (quality={quality}, language={language})"
                )

            result = await asyncio.to_thread(
                transcriber.transcribe_file,
                input_path,
                language,
                quality,
            )

            runtime = transcriber.status()
            summary = (
                f"Done.\n"
                f"Detected language: {result.language}\n"
                f"Quality mode: {quality}\n"
                f"Device: {runtime['active_device']}\n"
                f"Chunks used: {result.chunk_count}"
            )

            chunks = _split_for_telegram(result.text)
            if len(chunks) <= 8:
                await progress.edit_text(summary)
                for i, chunk in enumerate(chunks, start=1):
                    if len(chunks) == 1:
                        await message.reply_text(chunk)
                    else:
                        await message.reply_text(f"[Part {i}/{len(chunks)}]\n{chunk}")
            else:
                transcript_file = tmp_path / "transcript.txt"
                transcript_file.write_text(result.text, encoding="utf-8")
                await progress.edit_text(summary)
                await message.reply_document(
                    document=InputFile(str(transcript_file), filename="transcript.txt"),
                    caption="Transcript was long, sending as file.",
                )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process audio")
        await progress.edit_text(f"Error: {exc}")


def main() -> None:
    load_dotenv()
    _configure_cuda_paths()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in environment or .env file.")

    model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    target_chunk_mb = int(os.getenv("TARGET_CHUNK_MB", "20"))
    max_chunk_seconds = int(os.getenv("MAX_CHUNK_SECONDS", "900"))
    chunk_overlap_seconds = float(os.getenv("CHUNK_OVERLAP_SECONDS", "2.0"))
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "8"))
    best_of = int(os.getenv("WHISPER_BEST_OF", "8"))
    temperature = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
    vad_filter = _env_bool("WHISPER_VAD_FILTER", False)
    condition_on_previous_text = _env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)

    logger.info(
        "Initializing bot (model '%s' will load lazily on first transcription request).",
        model_size,
    )
    transcriber = SpeechTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        target_chunk_mb=target_chunk_mb,
        max_chunk_seconds=max_chunk_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )
    logger.info("Bot is starting.")

    app = Application.builder().token(token).build()
    app.bot_data["transcriber"] = transcriber

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("lang", set_language))
    app.add_handler(CommandHandler("quality", set_quality))
    app.add_handler(CommandHandler("gpu", gpu_status_command))
    app.add_handler(
        CallbackQueryHandler(
            settings_callback,
            pattern=r"^(lang:|quality:|settings:)",
        )
    )
    app.add_handler(
        MessageHandler(
            filters.AUDIO | filters.VOICE | filters.Document.AUDIO,
            transcribe_audio,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_instructions))

    # Python 3.14 no longer provides an implicit default event loop.
    # python-telegram-bot expects one when calling run_polling().
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
