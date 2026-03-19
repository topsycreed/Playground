from __future__ import annotations

import atexit
import asyncio
import io
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import threading
from tempfile import TemporaryDirectory
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.error import BadRequest, InvalidToken, NetworkError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
try:
    from yt_dlp import YoutubeDL
except ImportError:  # pragma: no cover - optional dependency at runtime
    YoutubeDL = None

from transcriber import SpeechTranscriber
from text_postprocessor import TextPostProcessor

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# Prevent token exposure in verbose HTTP request logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SUPPORTED_LANGUAGES = {"auto", "ru", "en"}
SUPPORTED_QUALITIES = {"fast", "balanced", "best"}
SUPPORTED_FORMATS = {"text", "dialog"}
SUPPORTED_DIARIZATION = {"auto", "nemo", "heuristic"}
SUPPORTED_SPEAKERS = {0, 2, 3, 4}
SUPPORTED_POSTPROCESS_MODELS = {"gemini", "whisper", "oos20"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}
SUPPORTED_TEXT_DEBUG_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".log",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".srt",
    ".vtt",
}
SUPPORTED_TEXT_DEBUG_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/x-yaml",
}

user_language: dict[int, str] = {}
user_quality: dict[int, str] = {}
user_format: dict[int, str] = {}
user_speakers: dict[int, int] = {}
user_postprocess_model: dict[int, str] = {}
local_bot_api_process: subprocess.Popen | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _read_env_value_from_file(name: str, env_file: Path) -> str:
    if not env_file.exists():
        return ""
    prefix = f"{name}="
    try:
        lines = env_file.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip().strip('"').strip("'")
        return value
    return ""


def _get_user_language(user_id: int) -> str:
    return user_language.get(user_id, "auto")


def _get_user_quality(user_id: int) -> str:
    return user_quality.get(user_id, "balanced")


def _get_user_format(user_id: int) -> str:
    return user_format.get(user_id, "dialog")


def _default_postprocess_model() -> str:
    raw = os.getenv("TEXT_POSTPROCESS_MODEL", "gemini").strip().lower()
    if raw not in SUPPORTED_POSTPROCESS_MODELS:
        return "gemini"
    return raw


def _get_user_postprocess_model(user_id: int) -> str:
    if user_id in user_postprocess_model:
        chosen = user_postprocess_model[user_id]
        if chosen in SUPPORTED_POSTPROCESS_MODELS:
            return chosen
    return _default_postprocess_model()


def _postprocess_model_label(value: str) -> str:
    mapping = {
        "gemini": "Google Gemini",
        "whisper": "Whisper",
        "oos20": "OpenAI oos-20",
    }
    return mapping.get(value, value)


def _default_speakers(transcriber: SpeechTranscriber | None) -> int:
    if transcriber is not None and transcriber.nemo_num_speakers in SUPPORTED_SPEAKERS:
        return transcriber.nemo_num_speakers
    try:
        env_value = int(os.getenv("NEMO_NUM_SPEAKERS", "0"))
    except ValueError:
        env_value = 0
    return env_value if env_value in SUPPORTED_SPEAKERS else 0


def _get_user_speakers(user_id: int, transcriber: SpeechTranscriber | None = None) -> int:
    if user_id in user_speakers:
        value = user_speakers[user_id]
        if value in SUPPORTED_SPEAKERS:
            return value
    return _default_speakers(transcriber)


def _find_dll_on_path(dll_name: str) -> str | None:
    for path_item in os.environ.get("PATH", "").split(os.pathsep):
        if not path_item:
            continue
        candidate = Path(path_item) / dll_name
        if candidate.exists():
            return str(candidate)
    return None


def _find_executable_on_path(executable_name: str) -> str | None:
    for path_item in os.environ.get("PATH", "").split(os.pathsep):
        if not path_item:
            continue
        candidate = Path(path_item) / executable_name
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
    auto_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64",
    ]
    _prepend_paths(user_paths + auto_paths)


def _configure_ffmpeg_paths() -> None:
    env_extra = os.getenv("FFMPEG_EXTRA_PATHS", "").strip()
    user_paths = [p.strip() for p in env_extra.split(";") if p.strip()] if env_extra else []
    auto_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
    ]

    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        for pkg_dir in winget_root.glob("Gyan.FFmpeg_*"):
            auto_paths.append(str(pkg_dir / "bin"))
            for nested_bin in pkg_dir.glob("*\\bin"):
                auto_paths.append(str(nested_bin))

    _prepend_paths(user_paths + auto_paths)

    ffmpeg = _find_executable_on_path("ffmpeg.exe")
    ffprobe = _find_executable_on_path("ffprobe.exe")
    if ffmpeg:
        os.environ.setdefault("FFMPEG_BINARY", ffmpeg)
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg)
        logger.info("FFmpeg detected: %s", ffmpeg)
    else:
        logger.warning("ffmpeg.exe not found in PATH. NeMo/pydub audio conversion may fail.")
    if ffprobe:
        logger.info("FFprobe detected: %s", ffprobe)


def _current_diarization_backend(transcriber: SpeechTranscriber | None) -> str:
    if transcriber is None:
        return os.getenv("DIARIZATION_BACKEND", "auto").strip().lower() or "auto"
    return transcriber.diarization_backend


def _settings_text(user_id: int, transcriber: SpeechTranscriber | None = None) -> str:
    diarization_backend = _current_diarization_backend(transcriber)
    speakers = _get_user_speakers(user_id, transcriber)
    speakers_label = "auto" if speakers == 0 else str(speakers)
    post_model = _get_user_postprocess_model(user_id)
    return (
        "Settings menu:\n"
        f"Language: {_get_user_language(user_id)}\n"
        f"Quality: {_get_user_quality(user_id)}\n"
        f"Format: {_get_user_format(user_id)}\n"
        f"Diarization: {diarization_backend}\n"
        f"Speakers: {speakers_label}\n\n"
        f"Post-process model: {_postprocess_model_label(post_model)}\n\n"
        "RU: Отправьте MP3/аудио/voice, MP4-видео или YouTube-ссылку, и я сделаю транскрипт.\n"
        "EN: Send MP3/audio/voice, MP4 video, or YouTube URL and I will transcribe it."
    )


def _settings_keyboard(
    user_id: int,
    transcriber: SpeechTranscriber | None = None,
) -> InlineKeyboardMarkup:
    lang = _get_user_language(user_id)
    quality = _get_user_quality(user_id)
    out_format = _get_user_format(user_id)
    diarization_backend = _current_diarization_backend(transcriber)
    speaker_mode = _get_user_speakers(user_id, transcriber)
    post_model = _get_user_postprocess_model(user_id)

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
                InlineKeyboardButton(mark(out_format == "text", "Text"), callback_data="format:text"),
                InlineKeyboardButton(mark(out_format == "dialog", "Dialog"), callback_data="format:dialog"),
            ],
            [
                InlineKeyboardButton(mark(diarization_backend == "auto", "Diar Auto"), callback_data="diar:auto"),
                InlineKeyboardButton(mark(diarization_backend == "nemo", "Diar NeMo"), callback_data="diar:nemo"),
                InlineKeyboardButton(mark(diarization_backend == "heuristic", "Diar Heuristic"), callback_data="diar:heuristic"),
            ],
            [
                InlineKeyboardButton(mark(speaker_mode == 0, "Spk Auto"), callback_data="spk:auto"),
                InlineKeyboardButton(mark(speaker_mode == 2, "Spk 2"), callback_data="spk:2"),
                InlineKeyboardButton(mark(speaker_mode == 3, "Spk 3"), callback_data="spk:3"),
                InlineKeyboardButton(mark(speaker_mode == 4, "Spk 4"), callback_data="spk:4"),
            ],
            [
                InlineKeyboardButton(mark(post_model == "gemini", "Google Gemini"), callback_data="model:gemini"),
                InlineKeyboardButton(mark(post_model == "whisper", "Whisper"), callback_data="model:whisper"),
                InlineKeyboardButton(mark(post_model == "oos20", "OpenAI oos-20"), callback_data="model:oos20"),
            ],
            [
                InlineKeyboardButton("Show current", callback_data="settings:show"),
                InlineKeyboardButton("Close menu", callback_data="settings:close"),
            ],
        ]
    )


async def _send_menu(
    message,
    user_id: int,
    transcriber: SpeechTranscriber | None = None,
) -> None:
    await message.reply_text(
        _settings_text(user_id, transcriber),
        reply_markup=_settings_keyboard(user_id, transcriber),
    )


async def _safe_edit_settings_menu(
    query,
    user_id: int,
    transcriber: SpeechTranscriber | None,
) -> None:
    await _safe_query_edit_message_text(
        query,
        _settings_text(user_id, transcriber),
        reply_markup=_settings_keyboard(user_id, transcriber),
    )


async def _safe_query_answer(
    query,
    text: str | None = None,
    *,
    show_alert: bool = False,
    retries: int = 2,
) -> bool:
    for attempt in range(1, max(1, retries) + 1):
        try:
            await query.answer(text, show_alert=show_alert)
            return True
        except BadRequest as exc:
            msg = str(exc).lower()
            # Callback may already be answered/expired; not fatal.
            if "query is too old" in msg or "query id is invalid" in msg:
                return False
            raise
        except (TimedOut, NetworkError) as exc:
            if attempt >= retries:
                logger.warning("query.answer failed after retries: %s", exc)
                return False
            await asyncio.sleep(0.35 * attempt)
    return False


async def _safe_query_edit_message_text(
    query,
    text: str,
    *,
    reply_markup=None,
    retries: int = 2,
) -> bool:
    for attempt in range(1, max(1, retries) + 1):
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
            return True
        except BadRequest as exc:
            msg = str(exc).lower()
            if "message is not modified" in msg:
                return False
            if "message to edit not found" in msg:
                return False
            raise
        except (TimedOut, NetworkError) as exc:
            if attempt >= retries:
                logger.warning("query.edit_message_text failed after retries: %s", exc)
                return False
            await asyncio.sleep(0.35 * attempt)
    return False


def _extract_file_info(update: Update) -> tuple[str, str, str]:
    message = update.effective_message
    if message is None:
        raise ValueError("No message to process.")

    if message.audio:
        ext = Path(message.audio.file_name or "audio.mp3").suffix or ".mp3"
        return message.audio.file_id, ext, "audio"
    if message.voice:
        return message.voice.file_id, ".ogg", "voice"
    if message.video:
        ext = Path(message.video.file_name or "video.mp4").suffix or ".mp4"
        return message.video.file_id, ext, "video"
    if message.document:
        default_name = "media.bin"
        mime = (message.document.mime_type or "").strip().lower()
        media_kind = "document"
        if mime.startswith("video/"):
            default_name = "video.mp4"
            media_kind = "video"
        elif mime.startswith("audio/"):
            default_name = "audio.bin"
            media_kind = "audio"
        ext = Path(message.document.file_name or default_name).suffix or Path(default_name).suffix or ".bin"
        if ext.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            media_kind = "video"
        return message.document.file_id, ext, media_kind

    raise ValueError("No supported audio/video payload found.")


def _extract_file_size(update: Update) -> int:
    message = update.effective_message
    if message is None:
        return 0
    if message.audio and message.audio.file_size:
        return int(message.audio.file_size)
    if message.voice and message.voice.file_size:
        return int(message.voice.file_size)
    if message.video and message.video.file_size:
        return int(message.video.file_size)
    if message.document and message.document.file_size:
        return int(message.document.file_size)
    return 0


def _extract_document_info(update: Update) -> tuple[str, str, int, str, str]:
    message = update.effective_message
    if message is None or message.document is None:
        raise ValueError("No document payload found.")
    doc = message.document
    name = (doc.file_name or "document.txt").strip() or "document.txt"
    ext = Path(name).suffix.lower()
    if not ext:
        ext = ".txt"
    size = int(doc.file_size or 0)
    mime_type = (doc.mime_type or "").strip().lower()
    return doc.file_id, ext, size, mime_type, name


def _is_supported_text_document(mime_type: str, extension: str) -> bool:
    ext = (extension or "").lower()
    if ext in SUPPORTED_TEXT_DEBUG_EXTENSIONS:
        return True
    mt = (mime_type or "").lower()
    if mt.startswith("text/"):
        return True
    if mt in SUPPORTED_TEXT_DEBUG_MIME_TYPES:
        return True
    return False


def _text_debug_max_bytes() -> int:
    raw = os.getenv("TEXT_DEBUG_MAX_MB", "50").strip()
    try:
        mb = int(raw)
    except ValueError:
        mb = 50
    mb = max(1, mb)
    return mb * 1024 * 1024


def _decode_text_payload(raw: bytes) -> tuple[str, str]:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1251",
        "windows-1251",
    ]
    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def _telegram_download_limit_bytes() -> int:
    raw = os.getenv("TELEGRAM_DOWNLOAD_LIMIT_MB", "20").strip()
    try:
        mb = int(raw)
    except ValueError:
        mb = 20
    mb = max(1, mb)
    return mb * 1024 * 1024


def _size_to_mb_text(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _find_youtube_url_in_text(text: str) -> str | None:
    if not text.strip():
        return None
    # Keep detection simple and robust for message text with extra words.
    url_pattern = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)
    for raw_url in url_pattern.findall(text):
        candidate = raw_url.rstrip(".,!?;:)]}>'\"")
        try:
            parsed = urllib_parse.urlsplit(candidate)
        except ValueError:
            continue
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        is_youtube = host in {"youtube.com", "m.youtube.com", "youtu.be"} or host.endswith(".youtube.com")
        if is_youtube:
            return candidate
    return None


def _download_audio_from_youtube(url: str, output_dir: Path) -> tuple[Path, str]:
    if YoutubeDL is None:
        raise RuntimeError("yt-dlp is not installed. Install dependency: pip install yt-dlp")

    output_template = str(output_dir / "youtube_audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": output_template,
        "restrictfilenames": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError("yt-dlp did not return media info.")
        if "entries" in info and info["entries"]:
            info = info["entries"][0]
        downloaded_path = Path(ydl.prepare_filename(info))

    if not downloaded_path.exists():
        candidates = sorted(output_dir.glob("youtube_audio.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("yt-dlp finished but downloaded audio file was not found.")
        downloaded_path = candidates[0]

    title = str(info.get("title") or "").strip() if isinstance(info, dict) else ""
    return downloaded_path, title


def _extract_audio_track_from_video(input_video: Path, output_audio: Path) -> None:
    ffmpeg_bin = os.getenv("FFMPEG_BINARY", "ffmpeg").strip() or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_audio),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-1200:]
        raise RuntimeError(f"ffmpeg audio extraction failed (code={proc.returncode}): {stderr_tail}")
    if not output_audio.exists() or output_audio.stat().st_size <= 0:
        raise RuntimeError("ffmpeg audio extraction produced empty output file.")


def _local_api_host() -> str:
    return os.getenv("TELEGRAM_LOCAL_API_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _local_api_port() -> int:
    raw = os.getenv("TELEGRAM_LOCAL_API_PORT", "8081").strip()
    try:
        port = int(raw)
    except ValueError:
        port = 8081
    return min(max(port, 1), 65535)


def _local_api_base_url() -> str:
    return f"http://{_local_api_host()}:{_local_api_port()}"


def _is_local_api_reachable(timeout_sec: float = 0.5) -> bool:
    try:
        with socket.create_connection((_local_api_host(), _local_api_port()), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _stop_local_bot_api_process() -> None:
    global local_bot_api_process
    process = local_bot_api_process
    if process is None:
        return
    if process.poll() is not None:
        local_bot_api_process = None
        return
    logger.info("Stopping local Telegram Bot API server (pid=%s)...", process.pid)
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
    local_bot_api_process = None


def _start_local_bot_api_if_needed() -> None:
    global local_bot_api_process
    if not _env_bool("TELEGRAM_LOCAL_MODE", False):
        return

    if _is_local_api_reachable():
        logger.info(
            "Using already running local Telegram Bot API at %s.",
            _local_api_base_url(),
        )
        return

    if not _env_bool("TELEGRAM_AUTO_START_LOCAL_API", False):
        raise RuntimeError(
            "TELEGRAM_LOCAL_MODE=true but local Telegram Bot API server is not reachable. "
            "Start it manually or set TELEGRAM_AUTO_START_LOCAL_API=true."
        )

    bot_api_bin = os.getenv("TELEGRAM_BOT_API_BIN", "").strip()
    if not bot_api_bin:
        raise RuntimeError("Set TELEGRAM_BOT_API_BIN to auto-start local Telegram Bot API server.")
    api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    if not api_id or not api_hash:
        raise RuntimeError("Set TELEGRAM_API_ID and TELEGRAM_API_HASH for local Telegram Bot API server.")

    data_dir_raw = os.getenv("TELEGRAM_LOCAL_API_DATA_DIR", ".telegram-bot-api-data").strip()
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        bot_api_bin,
        "--local",
        f"--api-id={api_id}",
        f"--api-hash={api_hash}",
        f"--http-port={_local_api_port()}",
        f"--dir={data_dir}",
    ]
    extra_args = os.getenv("TELEGRAM_BOT_API_EXTRA_ARGS", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args, posix=os.name != "nt"))

    logger.info("Starting local Telegram Bot API server...")
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    local_bot_api_process = process

    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        if _is_local_api_reachable(timeout_sec=0.8):
            logger.info(
                "Local Telegram Bot API is ready at %s (pid=%s).",
                _local_api_base_url(),
                process.pid,
            )
            return
        if process.poll() is not None:
            break
        time.sleep(0.4)

    _stop_local_bot_api_process()
    raise RuntimeError(
        "Failed to start local Telegram Bot API server automatically. "
        "Check TELEGRAM_BOT_API_BIN, TELEGRAM_API_ID, TELEGRAM_API_HASH."
    )


def _normalize_bot_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/bot"):
        return normalized
    return f"{normalized}/bot"


def _normalize_file_base_url(file_base_url: str) -> str:
    normalized = file_base_url.rstrip("/")
    if normalized.endswith("/file/bot"):
        return normalized
    if normalized.endswith("/file"):
        return f"{normalized}/bot"
    return f"{normalized}/file/bot"


def _prefer_loopback_ipv4(url: str) -> str:
    parsed = urllib_parse.urlsplit(url)
    hostname = parsed.hostname or ""
    if hostname.lower() != "localhost":
        return url
    port = f":{parsed.port}" if parsed.port else ""
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    netloc = f"{userinfo}127.0.0.1{port}"
    return urllib_parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _check_local_bot_api_get_me(
    token: str,
    bot_base_url: str,
    retries: int = 6,
    delay_seconds: float = 0.6,
) -> tuple[bool, str]:
    if not token:
        return False, "missing TELEGRAM_BOT_TOKEN"
    base = _normalize_bot_base_url(bot_base_url)
    get_me_url = f"{base}{token}/getMe"
    last_reason = "unknown error"
    for _ in range(max(1, retries)):
        try:
            with urllib_request.urlopen(get_me_url, timeout=4) as response:
                if response.status != 200:
                    last_reason = f"http {response.status}"
                    time.sleep(delay_seconds)
                    continue
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
                if bool(payload.get("ok")):
                    return True, "ok"
                last_reason = "response ok=false"
        except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_reason = f"{exc.__class__.__name__}: {exc}"
        time.sleep(delay_seconds)
    return False, last_reason


def _local_bot_api_data_roots() -> list[Path]:
    roots: list[Path] = []
    explicit_docker_root = os.getenv("TELEGRAM_LOCAL_DOCKER_DATA_DIR", "").strip()
    if explicit_docker_root:
        roots.append(Path(explicit_docker_root))

    local_data_dir = os.getenv("TELEGRAM_LOCAL_API_DATA_DIR", "").strip()
    if local_data_dir:
        local_data_path = Path(local_data_dir)
        if not local_data_path.is_absolute():
            local_data_path = Path.cwd() / local_data_path
        roots.append(local_data_path)

    roots.append(Path.cwd() / "tg-bot-api-data")
    roots.append(Path.cwd().parent / "tg-bot-api-data")

    deduped_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key in seen_roots:
            continue
        seen_roots.add(key)
        deduped_roots.append(root)
    return deduped_roots


async def _get_file_with_retries(
    bot,
    file_id: str,
    retries: int = 4,
    base_delay_seconds: float = 0.9,
):
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return await bot.get_file(file_id)
        except (TimedOut, NetworkError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            sleep_for = base_delay_seconds * attempt
            logger.warning(
                "get_file timeout/network error (attempt %s/%s): %s; retrying in %.1fs",
                attempt,
                retries,
                exc,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("get_file failed unexpectedly without exception")


def _resolve_local_bot_api_file_path(file_path: str | None) -> Path | None:
    if not file_path:
        return None
    raw = file_path.strip()
    if not raw:
        return None

    direct = Path(raw)
    if direct.exists() and direct.is_file():
        return direct

    docker_root_prefix = "/var/lib/telegram-bot-api/"
    colon_replacement = "\uf03a"

    deduped_roots = _local_bot_api_data_roots()

    relative_options: list[Path] = []
    normalized = raw.replace("\\", "/")
    if normalized.startswith(docker_root_prefix):
        rel = normalized[len(docker_root_prefix) :]
        if rel:
            parts = [part for part in rel.split("/") if part]
            relative_options.append(Path(*parts))
            translated = [part.replace(":", colon_replacement) for part in parts]
            if translated != parts:
                relative_options.append(Path(*translated))
    else:
        parts = [part for part in normalized.split("/") if part]
        if parts:
            relative_options.append(Path(*parts))
            translated = [part.replace(":", colon_replacement) for part in parts]
            if translated != parts:
                relative_options.append(Path(*translated))

    # Try direct relative match under root and under token directories.
    for root in deduped_roots:
        if not root.exists():
            continue
        for rel_path in relative_options:
            candidate = root / rel_path
            if candidate.exists() and candidate.is_file():
                return candidate
        try:
            token_dirs = [entry for entry in root.iterdir() if entry.is_dir()]
        except OSError:
            token_dirs = []
        for token_dir in token_dirs:
            for rel_path in relative_options:
                candidate = token_dir / rel_path
                if candidate.exists() and candidate.is_file():
                    return candidate

    # Fallback: search by basename and pick the freshest match.
    basename = Path(normalized).name
    if basename:
        basename_lower = basename.lower()
        newest_match: Path | None = None
        newest_mtime: float = -1.0
        for root in deduped_roots:
            if not root.exists():
                continue
            try:
                for candidate in root.rglob("*"):
                    if not candidate.is_file():
                        continue
                    if candidate.name.lower() != basename_lower:
                        continue
                    try:
                        mtime = candidate.stat().st_mtime
                    except OSError:
                        mtime = -1.0
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest_match = candidate
            except OSError:
                continue
        if newest_match is not None:
            return newest_match

    return None


def _resolve_recent_local_media_file(file_ext: str, expected_size: int = 0) -> Path | None:
    ext = (file_ext or "").lower()
    newest_match: Path | None = None
    newest_mtime: float = -1.0
    for root in _local_bot_api_data_roots():
        if not root.exists():
            continue
        try:
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                if ext and candidate.suffix.lower() != ext:
                    continue
                try:
                    stat = candidate.stat()
                except OSError:
                    continue
                if expected_size > 0 and stat.st_size != expected_size:
                    continue
                if stat.st_mtime > newest_mtime:
                    newest_mtime = stat.st_mtime
                    newest_match = candidate
        except OSError:
            continue
    return newest_match


def _local_direct_pickup_threshold_bytes() -> int:
    raw = os.getenv("TELEGRAM_LOCAL_DIRECT_PICKUP_MB", "40").strip()
    try:
        mb = int(raw)
    except ValueError:
        mb = 40
    mb = max(1, mb)
    return mb * 1024 * 1024


def _local_direct_pickup_wait_seconds() -> float:
    return max(1.0, _env_float("TELEGRAM_LOCAL_PICKUP_WAIT_SECONDS", 30.0))


async def _wait_for_recent_local_media_file(
    file_ext: str,
    expected_size: int,
    timeout_seconds: float,
) -> Path | None:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    while time.monotonic() < deadline:
        candidate = _resolve_recent_local_media_file(file_ext, expected_size=expected_size)
        if candidate is not None:
            return candidate
        await asyncio.sleep(0.7)
    return None


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


async def _send_text_or_file(
    message,
    text: str,
    *,
    filename: str,
    short_prefix: str = "",
    caption: str,
    force_file: bool = False,
) -> None:
    payload = text.strip()
    if not payload:
        payload = "(empty)"
    if not force_file:
        chunks = _split_for_telegram(payload)
        if len(chunks) <= 8:
            for idx, chunk in enumerate(chunks, start=1):
                if len(chunks) == 1:
                    body = chunk
                    if short_prefix:
                        body = f"{short_prefix}\n{chunk}"
                else:
                    body = f"[Part {idx}/{len(chunks)}]\n{chunk}"
                    if short_prefix and idx == 1:
                        body = f"{short_prefix}\n{body}"
                await message.reply_text(body)
            return

    payload_bytes = io.BytesIO(payload.encode("utf-8"))
    payload_bytes.seek(0)
    await message.reply_document(
        document=InputFile(payload_bytes, filename=filename),
        caption=caption,
    )


def _render_progress_bar(done: int, total: int, width: int = 18) -> str:
    if total <= 0:
        return "[------------------] 0%"
    clamped_done = max(0, min(done, total))
    ratio = clamped_done / total
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {int(ratio * 100):d}%"


def _build_progress_text(state: dict[str, object], elapsed_seconds: float) -> str:
    stage = str(state.get("stage", "transcribing"))
    msg = str(state.get("message", "Processing audio..."))
    done_chunks = int(state.get("done_chunks", 0) or 0)
    total_chunks = int(state.get("total_chunks", 0) or 0)

    lines = [msg]
    if total_chunks > 0:
        lines.append(f"{_render_progress_bar(done_chunks, total_chunks)}  ({done_chunks}/{total_chunks} chunks)")
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    lines.append(f"Elapsed: {minutes:02d}:{seconds:02d}")
    if stage == "model_loading":
        lines.append("Stage: loading model")
    elif stage == "diarization":
        lines.append("Stage: speaker diarization")
    elif stage == "finalizing":
        lines.append("Stage: finalizing")
    else:
        lines.append("Stage: transcription")
    return "\n".join(lines)


async def _progress_message_updater(
    progress_message,
    state: dict[str, object],
    state_lock: threading.Lock,
    stop_event: asyncio.Event,
) -> None:
    started = time.monotonic()
    last_text = ""
    while not stop_event.is_set():
        await asyncio.sleep(4)
        with state_lock:
            snapshot = dict(state)
        text = _build_progress_text(snapshot, time.monotonic() - started)
        if text == last_text:
            continue
        try:
            await progress_message.edit_text(text)
            last_text = text
        except BadRequest as exc:
            if "message is not modified" in str(exc).lower():
                continue
            logger.debug("Progress edit ignored: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Progress updater failed: %s", exc)


async def _finalize_and_send_transcription_result(
    *,
    message,
    progress,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    transcriber: SpeechTranscriber,
    result,
    quality: str,
    output_format: str,
    speaker_mode: int,
    post_model: str,
    state_lock: threading.Lock | None = None,
    progress_state: dict[str, object] | None = None,
) -> None:
    postprocess_note = ""
    summary_note = "\nSummary: not generated"
    summary_text = ""
    postprocessor: TextPostProcessor | None = context.application.bot_data.get("postprocessor")
    if postprocessor is not None and postprocessor.enabled:
        if state_lock is not None and progress_state is not None:
            with state_lock:
                progress_state.update(
                    {
                        "stage": "finalizing",
                        "message": "Post-processing transcript text...",
                    }
                )
        processed_text, post_report = await asyncio.to_thread(
            postprocessor.process_text,
            result.text,
            result.language,
            output_format,
            post_model,
        )
        result.text = processed_text
        if post_report.applied:
            rename_info = ""
            if post_report.renamed_speakers:
                rename_info = f", renamed speakers: {len(post_report.renamed_speakers)}"
            postprocess_note = (
                f"\nPost-processing model: {_postprocess_model_label(post_model)}"
                f"\nPost-processing: {post_report.method}{rename_info}"
            )
        elif post_report.error:
            postprocess_note = (
                f"\nPost-processing model: {_postprocess_model_label(post_model)}"
                f"\nPost-processing fallback: {post_report.method} ({post_report.error})"
            )
        else:
            postprocess_note = (
                f"\nPost-processing model: {_postprocess_model_label(post_model)}"
                f"\nPost-processing: {post_report.method}"
            )

        if state_lock is not None and progress_state is not None:
            with state_lock:
                progress_state.update(
                    {
                        "stage": "finalizing",
                        "message": "Generating summary...",
                    }
                )
        summary_text, summary_report = await asyncio.to_thread(
            postprocessor.summarize_text,
            result.text,
            result.language,
            post_model,
        )
        summary_note = f"\nSummary: {summary_report.method}"
        if summary_report.error:
            summary_note = f"\nSummary: {summary_report.method} ({summary_report.error})"

    runtime = transcriber.status()
    format_note = output_format
    if output_format == "dialog" and result.speaker_count <= 1:
        format_note = "dialog (auto-switched to plain text: 1 speaker)"
    nemo_note = ""
    if runtime.get("nemo_available") is False:
        reason = runtime.get("nemo_reason") or "NeMo is unavailable."
        nemo_note = f"\nNeMo status: unavailable ({reason})"
    summary = (
        f"Done.\n"
        f"Detected language: {result.language}\n"
        f"Speakers detected: {result.speaker_count}\n"
        f"Diarization backend: {result.diarization_backend}\n"
        f"Speaker mode: {'auto' if speaker_mode == 0 else speaker_mode}\n"
        f"Quality mode: {quality}\n"
        f"Format: {format_note}\n"
        f"Device: {runtime['active_device']}\n"
        f"Chunks used: {result.chunk_count}"
        f"{nemo_note}"
        f"{postprocess_note}"
        f"{summary_note}"
    )

    await progress.edit_text(summary)
    await _send_text_or_file(
        message,
        result.text,
        filename="transcript.txt",
        caption="Transcript attached as file.",
        force_file=True,
    )
    if summary_text.strip():
        await _send_text_or_file(
            message,
            summary_text,
            filename="summary.txt",
            short_prefix="Summary:",
            caption="Summary was long, sending as file.",
        )


def _format_gpu_status(transcriber: SpeechTranscriber) -> str:
    status = transcriber.status()
    cublas = _find_dll_on_path("cublas64_12.dll") or "not found"
    cudnn = _find_dll_on_path("cudnn64_9.dll") or "not found"
    ffmpeg = _find_executable_on_path("ffmpeg.exe") or "not found"
    ffprobe = _find_executable_on_path("ffprobe.exe") or "not found"
    nemo_available = status.get("nemo_available")
    if nemo_available is None:
        nemo_line = "NeMo available: n/a (backend disabled)"
    else:
        nemo_line = f"NeMo available: {'yes' if nemo_available else 'no'}"
        if not nemo_available and status.get("nemo_reason"):
            nemo_line = f"{nemo_line}\nNeMo note: {status['nemo_reason']}"
    return (
        "Runtime status:\n"
        f"Model: {status['model_size']}\n"
        f"Requested device: {status['requested_device']}\n"
        f"Active device: {status['active_device']}\n"
        f"Requested compute: {status['requested_compute_type']}\n"
        f"Active compute: {status['active_compute_type']}\n"
        f"Diarization backend: {status['diarization_backend']}\n"
        f"Default NeMo speakers: {status['nemo_num_speakers']}\n"
        f"Model loaded: {'yes' if status['model_loaded'] else 'no'}\n"
        f"Loading now: {'yes' if status['loading'] else 'no'}\n"
        f"{nemo_line}\n"
        f"cublas64_12.dll: {cublas}\n"
        f"cudnn64_9.dll: {cudnn}\n"
        f"ffmpeg.exe: {ffmpeg}\n"
        f"ffprobe.exe: {ffprobe}"
    )


def _format_llm_status(status: dict[str, object]) -> str:
    enabled = bool(status.get("enabled"))
    provider = str(status.get("provider", "unknown"))
    base_url = str(status.get("base_url", ""))
    api_base_url = str(status.get("api_base_url", "")).strip()
    prompts_file = str(status.get("prompts_file", "")).strip()
    normalization_entries = int(status.get("normalization_entries", 0) or 0)
    timeout_sec = float(status.get("timeout_sec", 0) or 0)
    request_retries = int(status.get("request_retries", 0) or 0)
    chunk_chars = int(status.get("chunk_chars", 0) or 0)
    summary_chunk_chars = int(status.get("summary_chunk_chars", 0) or 0)
    configured_model = str(status.get("configured_model", "(auto)"))
    effective_model = str(status.get("effective_model", "")) or "(not resolved)"
    gemini_model = str(status.get("gemini_model", "")).strip() or "(not set)"
    gemini_api_key_set = bool(status.get("gemini_api_key_set"))
    gemini_api_keys_count = int(status.get("gemini_api_keys_count", 0) or 0)
    gemini_timeout_sec = float(status.get("gemini_timeout_sec", 0) or 0)
    gemini_fallback_model = str(status.get("gemini_fallback_model", "whisper") or "whisper")
    available = bool(status.get("available"))
    models = status.get("models") or []
    error = str(status.get("error", "")).strip()

    lines = [
        "LLM post-processing status:",
        f"Enabled: {'yes' if enabled else 'no'}",
        f"Provider: {provider}",
        f"Base URL: {base_url}",
        f"Resolved API endpoint: {api_base_url or '(not resolved yet)'}",
        f"Prompts file: {prompts_file or '(default)'}",
        f"Normalization entries: {normalization_entries}",
        f"Timeout (sec): {timeout_sec:g}",
        f"Retries: {request_retries}",
        f"Cleanup chunk chars: {chunk_chars}",
        f"Summary chunk chars: {summary_chunk_chars}",
        f"Configured model: {configured_model}",
        f"Effective model: {effective_model}",
        f"Gemini model: {gemini_model}",
        f"Gemini key configured: {'yes' if gemini_api_key_set else 'no'}",
        f"Gemini keys count: {gemini_api_keys_count}",
        f"Gemini timeout (sec): {gemini_timeout_sec:g}",
        f"Gemini fallback model: {gemini_fallback_model}",
        f"LM Studio reachable: {'yes' if available else 'no'}",
    ]
    if isinstance(models, list) and models:
        preview = ", ".join(str(m) for m in models[:4])
        if len(models) > 4:
            preview += f", ... (+{len(models) - 4} more)"
        lines.append(f"Models: {preview}")
    if error:
        lines.append(f"Note: {error}")
    lines.append(
        "Hint: configure GEMINI_API_KEYS for Google Gemini free tier "
        "(single key or multiple keys separated by comma/semicolon/newline). "
        "Set /model gemini|whisper|oos20 in bot settings."
    )
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    await message.reply_text(
        "Send me an MP3/audio file, MP4 video, or YouTube URL and I will convert speech to text.\n"
        "Use /help for interactive settings menu.\n"
        "Send a text document (.txt/.md/.log/.json) to run debug cleanup+summary mode.\n"
        "Use /gpu to check CUDA/CPU runtime status.\n"
        "Use /llm to check LM Studio post-processing status.\n"
        "Use /model gemini|whisper|oos20 to choose post-process model.\n"
        "Use /diar auto|nemo|heuristic to choose diarization backend.\n"
        "Use /speakers auto|2|3|4 to guide speaker count.\n"
        "Default output format is dialog (User1/User2)."
    )
    await _send_menu(message, user.id, transcriber)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    await message.reply_text(
        "Commands:\n"
        "/help - open settings menu\n"
        "/settings - open settings menu\n"
        "/lang auto|ru|en\n"
        "/quality fast|balanced|best\n"
        "/format text|dialog\n"
        "/model gemini|whisper|oos20\n"
        "/diar auto|nemo|heuristic\n"
        "/speakers auto|2|3|4\n"
        "/gpu - show runtime device status\n\n"
        "/llm - show LM Studio post-processing status\n\n"
        "Send audio/video as voice, audio, video, file attachment, or YouTube URL.\n"
        "Debug text mode: send text file (.txt/.md/.log/.json) for cleanup + summary."
    )
    await _send_menu(message, user.id, transcriber)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    await _send_menu(message, user.id, transcriber)


async def transcribe_youtube_url(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    youtube_url: str,
) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber = context.application.bot_data["transcriber"]
    language = _get_user_language(user.id)
    quality = _get_user_quality(user.id)
    output_format = _get_user_format(user.id)
    speaker_mode = _get_user_speakers(user.id, transcriber)
    post_model = _get_user_postprocess_model(user.id)

    progress = await message.reply_text("YouTube mode: downloading audio...")
    state_lock = threading.Lock()
    progress_state: dict[str, object] = {
        "stage": "download",
        "message": "Downloading audio from YouTube...",
        "done_chunks": 0,
        "total_chunks": 0,
    }
    stop_event = asyncio.Event()
    updater_task: asyncio.Task | None = None
    try:
        if YoutubeDL is None:
            await progress.edit_text(
                "RU:\n"
                "Поддержка YouTube не установлена: нужен пакет `yt-dlp`.\n"
                "Установите: pip install yt-dlp\n\n"
                "EN:\n"
                "YouTube support is not installed: `yt-dlp` package is required.\n"
                "Install: pip install yt-dlp"
            )
            return

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            media_path, video_title = await asyncio.to_thread(
                _download_audio_from_youtube,
                youtube_url,
                tmp_path,
            )
            logger.info(
                "YouTube audio downloaded: url=%s title=%s path=%s size=%s",
                youtube_url,
                video_title or "(unknown)",
                media_path,
                _size_to_mb_text(media_path.stat().st_size),
            )

            transcribe_input_path = media_path
            if media_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                with state_lock:
                    progress_state.update(
                        {
                            "stage": "finalizing",
                            "message": "Extracting audio track from downloaded video...",
                        }
                    )
                await progress.edit_text("Extracting audio track from downloaded video...")
                extracted_audio_path = tmp_path / "youtube_extracted.wav"
                await asyncio.to_thread(
                    _extract_audio_track_from_video,
                    media_path,
                    extracted_audio_path,
                )
                transcribe_input_path = extracted_audio_path

            with state_lock:
                progress_state.update(
                    {
                        "stage": "model_loading" if not transcriber.is_model_loaded() else "transcribing",
                        "message": (
                            "Preparing model for first run..."
                            if not transcriber.is_model_loaded()
                            else "Transcription started..."
                        ),
                        "done_chunks": 0,
                        "total_chunks": 0,
                    }
                )
            title_note = f" title={video_title}" if video_title else ""
            await progress.edit_text(
                f"Processing YouTube audio...{title_note} (quality={quality}, language={language}, format={output_format})"
            )
            updater_task = asyncio.create_task(
                _progress_message_updater(progress, progress_state, state_lock, stop_event)
            )

            def _progress_callback(payload: dict[str, object]) -> None:
                with state_lock:
                    progress_state.update(payload)

            result = await asyncio.to_thread(
                transcriber.transcribe_file,
                transcribe_input_path,
                language,
                quality,
                output_format,
                speaker_mode,
                _progress_callback,
            )
            stop_event.set()
            if updater_task is not None:
                await updater_task

            await _finalize_and_send_transcription_result(
                message=message,
                progress=progress,
                context=context,
                user_id=user.id,
                transcriber=transcriber,
                result=result,
                quality=quality,
                output_format=output_format,
                speaker_mode=speaker_mode,
                post_model=post_model,
                state_lock=state_lock,
                progress_state=progress_state,
            )
    except Exception as exc:  # pylint: disable=broad-except
        stop_event.set()
        if updater_task is not None:
            await updater_task
        logger.exception("Failed to process YouTube URL: %s", youtube_url)
        await progress.edit_text(f"Error: {exc}")


async def text_instructions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    text_payload = (message.text or "").strip()
    youtube_url = _find_youtube_url_in_text(text_payload)
    if youtube_url:
        await transcribe_youtube_url(update, context, youtube_url)
        return

    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    await message.reply_text(
        "RU:\n"
        "Я распознаю речь из аудио/видео файлов.\n"
        "Пожалуйста, отправьте MP3/аудио/voice-сообщение или MP4-видео.\n"
        "Или отправьте ссылку на YouTube видео.\n"
        "Для debug-режима можно отправить текстовый файл (.txt/.md/.log/.json): "
        "я отдельно сделаю очистку и summary.\n\n"
        "EN:\n"
        "I transcribe speech from audio/video files.\n"
        "Please send an MP3/audio/voice message or MP4 video.\n"
        "Or send a YouTube video URL.\n"
        "For debug mode you can send a text file (.txt/.md/.log/.json): "
        "I will run separate cleanup and summary.\n\n"
        "Use /help for settings."
    )
    await _send_menu(message, user.id, transcriber)


async def debug_text_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None or message.document is None:
        return

    file_id, file_ext, file_size, mime_type, file_name = _extract_document_info(update)
    if not _is_supported_text_document(mime_type, file_ext):
        await message.reply_text(
            "RU:\n"
            "Это не текстовый файл для debug-обработки.\n"
            "Поддерживаются: .txt .md .log .csv .json .yaml .xml .srt .vtt\n\n"
            "EN:\n"
            "This is not a supported text file for debug processing.\n"
            "Supported: .txt .md .log .csv .json .yaml .xml .srt .vtt"
        )
        return

    max_size = _text_debug_max_bytes()
    if file_size > 0 and file_size > max_size:
        await message.reply_text(
            "RU:\n"
            f"Текстовый файл слишком большой для debug-режима "
            f"({_size_to_mb_text(file_size)} > {_size_to_mb_text(max_size)}).\n"
            "Уменьшите файл или поднимите лимит через TEXT_DEBUG_MAX_MB.\n\n"
            "EN:\n"
            f"Text file is too large for debug mode "
            f"({_size_to_mb_text(file_size)} > {_size_to_mb_text(max_size)}).\n"
            "Reduce file size or increase TEXT_DEBUG_MAX_MB."
        )
        return

    logger.info(
        "Debug text mode started: user_id=%s file=%s size=%s mime=%s ext=%s",
        user.id,
        file_name,
        file_size,
        mime_type,
        file_ext,
    )
    progress = await message.reply_text("Debug text mode: downloading document...")
    try:
        bot_is_local_mode = bool(getattr(context.bot, "local_mode", False))
        tg_file = None
        direct_source: Path | None = None

        if bot_is_local_mode and file_size > 0 and file_size >= _local_direct_pickup_threshold_bytes():
            direct_source = await _wait_for_recent_local_media_file(
                file_ext=file_ext,
                expected_size=file_size,
                timeout_seconds=_local_direct_pickup_wait_seconds(),
            )

        if direct_source is None:
            try:
                tg_file = await _get_file_with_retries(context.bot, file_id)
            except (TimedOut, NetworkError) as exc:
                if bot_is_local_mode:
                    direct_source = await _wait_for_recent_local_media_file(
                        file_ext=file_ext,
                        expected_size=file_size,
                        timeout_seconds=_local_direct_pickup_wait_seconds(),
                    )
                    if direct_source is None:
                        raise RuntimeError(
                            "Timeout while loading text file from Telegram/local storage."
                        ) from exc
                else:
                    raise

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / f"debug_input{file_ext}"

            if direct_source is not None:
                shutil.copyfile(direct_source, input_path)
            elif tg_file is not None:
                try:
                    await tg_file.download_to_drive(custom_path=str(input_path))
                except (InvalidToken, BadRequest) as exc:
                    msg_lower = str(exc).lower()
                    fallback_source = None
                    if bot_is_local_mode and ("not found" in msg_lower or "invalid token" in msg_lower):
                        fallback_source = _resolve_local_bot_api_file_path(tg_file.file_path)
                        if fallback_source is None:
                            fallback_source = await _wait_for_recent_local_media_file(
                                file_ext=file_ext,
                                expected_size=file_size,
                                timeout_seconds=_local_direct_pickup_wait_seconds(),
                            )
                    if fallback_source is None:
                        raise
                    shutil.copyfile(fallback_source, input_path)
            else:
                raise RuntimeError("No source available to download text file.")

            raw_payload = input_path.read_bytes()
            logger.info(
                "Debug text mode: download finished for user_id=%s file=%s bytes=%s",
                user.id,
                file_name,
                len(raw_payload),
            )

        await progress.edit_text("Debug text mode: cleaning and summarizing...")
        raw_text, decoded_as = _decode_text_payload(raw_payload)
        raw_text = raw_text.replace("\x00", "").strip()
        logger.info(
            "Debug text mode: decoded file=%s using=%s chars=%s",
            file_name,
            decoded_as,
            len(raw_text),
        )
        if not raw_text:
            await progress.edit_text("Debug text mode: file is empty after decoding.")
            return

        language = _get_user_language(user.id)
        post_model = _get_user_postprocess_model(user.id)
        postprocessor: TextPostProcessor | None = context.application.bot_data.get("postprocessor")
        if postprocessor is None:
            postprocessor = TextPostProcessor()

        process_started_at = time.perf_counter()
        cleaned_text, summary_text, debug_report = await asyncio.to_thread(
            postprocessor.process_debug_text,
            raw_text,
            language,
            post_model,
        )
        logger.info(
            "Debug text mode finished: user_id=%s file=%s model=%s cleanup=%s summary=%s elapsed=%.1fs cleaned_chars=%s summary_chars=%s",
            user.id,
            file_name,
            post_model,
            debug_report.cleanup_method,
            debug_report.summary_method,
            time.perf_counter() - process_started_at,
            len(cleaned_text),
            len(summary_text),
        )

        note = (
            "Debug text mode complete.\n"
            f"File: {file_name}\n"
            f"Decoded as: {decoded_as}\n"
            f"Model: {_postprocess_model_label(post_model)}\n"
            f"Cleanup: {debug_report.cleanup_method}\n"
            f"Summary: {debug_report.summary_method}"
        )
        if debug_report.error:
            note += f"\nNote: {debug_report.error}"
        await progress.edit_text(note)

        await _send_text_or_file(
            message,
            summary_text,
            filename="summary.txt",
            short_prefix="Summary:",
            caption="Summary was long, sending as file.",
        )
        await _send_text_or_file(
            message,
            cleaned_text,
            filename="cleaned_text.txt",
            short_prefix="Cleaned text:",
            caption="Cleaned text was long, sending as file.",
        )

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process text document in debug mode")
        await progress.edit_text(f"Debug text mode error: {exc}")


async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    if not context.args:
        await message.reply_text(f"Current language mode: {_get_user_language(user.id)}")
        return
    chosen = context.args[0].strip().lower()
    if chosen not in SUPPORTED_LANGUAGES:
        await message.reply_text("Use: /lang auto | /lang ru | /lang en")
        return
    user_language[user.id] = chosen
    await message.reply_text(f"Language mode set to: {chosen}")
    await _send_menu(message, user.id, transcriber)


async def set_quality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    if not context.args:
        await message.reply_text(f"Current quality mode: {_get_user_quality(user.id)}")
        return
    chosen = context.args[0].strip().lower()
    if chosen not in SUPPORTED_QUALITIES:
        await message.reply_text("Use: /quality fast | /quality balanced | /quality best")
        return
    user_quality[user.id] = chosen
    await message.reply_text(f"Quality mode set to: {chosen}")
    await _send_menu(message, user.id, transcriber)


async def set_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    if not context.args:
        await message.reply_text(f"Current output format: {_get_user_format(user.id)}")
        return
    chosen = context.args[0].strip().lower()
    if chosen not in SUPPORTED_FORMATS:
        await message.reply_text("Use: /format text | /format dialog")
        return
    user_format[user.id] = chosen
    await message.reply_text(f"Output format set to: {chosen}")
    await _send_menu(message, user.id, transcriber)


async def set_postprocess_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    if not context.args:
        current = _get_user_postprocess_model(user.id)
        await message.reply_text(f"Current post-process model: {_postprocess_model_label(current)}")
        return
    chosen = context.args[0].strip().lower()
    if chosen not in SUPPORTED_POSTPROCESS_MODELS:
        await message.reply_text("Use: /model gemini | /model whisper | /model oos20")
        return
    user_postprocess_model[user.id] = chosen
    await message.reply_text(f"Post-process model set to: {_postprocess_model_label(chosen)}")
    await _send_menu(message, user.id, transcriber)


async def set_diarization_backend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    if transcriber is None:
        await message.reply_text("Transcriber is not initialized yet.")
        return

    if not context.args:
        await message.reply_text(f"Current diarization backend: {transcriber.diarization_backend}")
        await _send_menu(message, user.id, transcriber)
        return

    chosen = context.args[0].strip().lower()
    if chosen not in SUPPORTED_DIARIZATION:
        await message.reply_text("Use: /diar auto | /diar nemo | /diar heuristic")
        return

    try:
        transcriber.set_diarization_backend(chosen)
    except ValueError:
        await message.reply_text("Unsupported diarization backend.")
        return

    await message.reply_text(f"Diarization backend set to: {chosen}")
    await _send_menu(message, user.id, transcriber)


async def set_speakers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    current = _get_user_speakers(user.id, transcriber)
    if not context.args:
        current_label = "auto" if current == 0 else str(current)
        await message.reply_text(f"Current speaker mode: {current_label}")
        await _send_menu(message, user.id, transcriber)
        return

    raw = context.args[0].strip().lower()
    if raw == "auto":
        chosen = 0
    else:
        try:
            chosen = int(raw)
        except ValueError:
            chosen = -1

    if chosen not in SUPPORTED_SPEAKERS:
        await message.reply_text("Use: /speakers auto | /speakers 2 | /speakers 3 | /speakers 4")
        return

    user_speakers[user.id] = chosen
    chosen_label = "auto" if chosen == 0 else str(chosen)
    await message.reply_text(f"Speaker mode set to: {chosen_label}")
    await _send_menu(message, user.id, transcriber)


async def gpu_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    transcriber: SpeechTranscriber = context.application.bot_data.get("transcriber")
    if transcriber is None:
        await message.reply_text("Transcriber is not initialized yet.")
        return
    await message.reply_text(_format_gpu_status(transcriber))


async def llm_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    postprocessor: TextPostProcessor | None = context.application.bot_data.get("postprocessor")
    if postprocessor is None:
        await message.reply_text("LLM post-processor is not initialized.")
        return
    status = await asyncio.to_thread(postprocessor.runtime_status)
    await message.reply_text(_format_llm_status(status))


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    if query is None or user is None:
        return

    transcriber: SpeechTranscriber | None = context.application.bot_data.get("transcriber")
    data = query.data or ""
    if data.startswith("lang:"):
        chosen = data.split(":", 1)[1]
        if chosen in SUPPORTED_LANGUAGES:
            user_language[user.id] = chosen
            await _safe_query_answer(query, f"Language: {chosen}")
        else:
            await _safe_query_answer(query, "Unsupported language", show_alert=True)
        await _safe_edit_settings_menu(query, user.id, transcriber)
        return

    if data.startswith("quality:"):
        chosen = data.split(":", 1)[1]
        if chosen in SUPPORTED_QUALITIES:
            user_quality[user.id] = chosen
            await _safe_query_answer(query, f"Quality: {chosen}")
        else:
            await _safe_query_answer(query, "Unsupported quality", show_alert=True)
        await _safe_edit_settings_menu(query, user.id, transcriber)
        return

    if data.startswith("format:"):
        chosen = data.split(":", 1)[1]
        if chosen in SUPPORTED_FORMATS:
            user_format[user.id] = chosen
            await _safe_query_answer(query, f"Format: {chosen}")
        else:
            await _safe_query_answer(query, "Unsupported format", show_alert=True)
        await _safe_edit_settings_menu(query, user.id, transcriber)
        return

    if data.startswith("model:"):
        chosen = data.split(":", 1)[1].strip().lower()
        if chosen in SUPPORTED_POSTPROCESS_MODELS:
            user_postprocess_model[user.id] = chosen
            await _safe_query_answer(query, f"Model: {_postprocess_model_label(chosen)}")
        else:
            await _safe_query_answer(query, "Unsupported model", show_alert=True)
        await _safe_edit_settings_menu(query, user.id, transcriber)
        return

    if data.startswith("diar:"):
        chosen = data.split(":", 1)[1]
        if chosen not in SUPPORTED_DIARIZATION:
            await _safe_query_answer(query, "Unsupported backend", show_alert=True)
            return
        if transcriber is None:
            await _safe_query_answer(query, "Transcriber not ready", show_alert=True)
            return
        try:
            transcriber.set_diarization_backend(chosen)
        except ValueError:
            await _safe_query_answer(query, "Unsupported backend", show_alert=True)
            return
        await _safe_query_answer(query, f"Diarization: {chosen}")
        await _safe_edit_settings_menu(query, user.id, transcriber)
        return

    if data.startswith("spk:"):
        raw = data.split(":", 1)[1].strip().lower()
        if raw == "auto":
            chosen = 0
        else:
            try:
                chosen = int(raw)
            except ValueError:
                chosen = -1
        if chosen not in SUPPORTED_SPEAKERS:
            await _safe_query_answer(query, "Unsupported speaker mode", show_alert=True)
            return
        user_speakers[user.id] = chosen
        chosen_label = "auto" if chosen == 0 else str(chosen)
        await _safe_query_answer(query, f"Speakers: {chosen_label}")
        await _safe_edit_settings_menu(query, user.id, transcriber)
        return

    if data == "settings:show":
        await _safe_query_answer(query, "Updated")
        await _safe_edit_settings_menu(query, user.id, transcriber)
        return

    if data == "settings:close":
        await _safe_query_answer(query, "Closed")
        await _safe_query_edit_message_text(query, "Menu closed. Use /help or /settings to open it again.")
        return

    await _safe_query_answer(query, "Unknown action")


async def transcribe_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    transcriber: SpeechTranscriber = context.application.bot_data["transcriber"]
    language = _get_user_language(user.id)
    quality = _get_user_quality(user.id)
    output_format = _get_user_format(user.id)
    speaker_mode = _get_user_speakers(user.id, transcriber)
    post_model = _get_user_postprocess_model(user.id)

    progress = await message.reply_text("Downloading audio...")
    state_lock = threading.Lock()
    progress_state: dict[str, object] = {
        "stage": "download",
        "message": "Downloading audio...",
        "done_chunks": 0,
        "total_chunks": 0,
    }
    stop_event = asyncio.Event()
    updater_task: asyncio.Task | None = None
    try:
        bot_is_local_mode = bool(getattr(context.bot, "local_mode", False))
        file_size = _extract_file_size(update)
        limit_bytes = _telegram_download_limit_bytes()
        if (not bot_is_local_mode) and file_size > 0 and file_size > limit_bytes:
            await progress.edit_text(
                "RU:\n"
                f"Файл слишком большой для загрузки через Telegram Bot API "
                f"({_size_to_mb_text(file_size)} > {_size_to_mb_text(limit_bytes)}).\n"
                "Пожалуйста, отправьте более маленький файл (или предварительно разделите аудио).\n\n"
                "EN:\n"
                f"File is too large for Telegram Bot API download "
                f"({_size_to_mb_text(file_size)} > {_size_to_mb_text(limit_bytes)}).\n"
                "Please send a smaller file (or split audio first)."
            )
            return

        file_id, file_ext, media_kind = _extract_file_info(update)
        tg_file = None
        direct_source: Path | None = None

        # For big local files, prefer direct pickup from local Bot API storage.
        if bot_is_local_mode and file_size > 0 and file_size >= _local_direct_pickup_threshold_bytes():
            direct_source = await _wait_for_recent_local_media_file(
                file_ext=file_ext,
                expected_size=file_size,
                timeout_seconds=_local_direct_pickup_wait_seconds(),
            )
            if direct_source is not None:
                logger.info(
                    "Using direct local storage source for large file: %s (%s)",
                    direct_source,
                    _size_to_mb_text(file_size),
                )

        if direct_source is None:
            try:
                tg_file = await _get_file_with_retries(context.bot, file_id)
            except BadRequest as exc:
                if "file is too big" in str(exc).lower():
                    if bot_is_local_mode:
                        await progress.edit_text(
                            "RU:\n"
                            "Локальный Bot API ответил: File is too big.\n"
                            "Обычно это значит, что сервер запущен без local mode.\n"
                            "Перезапустите контейнер с TELEGRAM_LOCAL=1 (или бинарник с --local).\n\n"
                            "EN:\n"
                            "Local Bot API returned: File is too big.\n"
                            "Usually this means the server is running without local mode.\n"
                            "Restart container with TELEGRAM_LOCAL=1 (or binary with --local)."
                        )
                    else:
                        await progress.edit_text(
                            "RU:\n"
                            "Telegram вернул ошибку: файл слишком большой для скачивания ботом.\n"
                            "Отправьте файл меньше лимита Bot API или разделите аудио перед отправкой.\n\n"
                            "EN:\n"
                            "Telegram returned: file is too big for bot download.\n"
                            "Send a smaller file or split audio before upload."
                        )
                    return
                raise
            except (TimedOut, NetworkError) as exc:
                if bot_is_local_mode:
                    direct_source = await _wait_for_recent_local_media_file(
                        file_ext=file_ext,
                        expected_size=file_size,
                        timeout_seconds=_local_direct_pickup_wait_seconds(),
                    )
                    if direct_source is not None:
                        logger.warning(
                            "get_file failed (%s), using local recent-file fallback: %s",
                            exc,
                            direct_source,
                        )
                    else:
                        await progress.edit_text(
                            "RU:\n"
                            "Таймаут при обращении к Telegram Bot API (getFile), "
                            "и файл не найден в локальном storage.\n"
                            "Проверьте Docker volume и попробуйте ещё раз.\n\n"
                            "EN:\n"
                            "Timeout while calling Telegram Bot API (getFile), "
                            "and file was not found in local storage fallback.\n"
                            "Check Docker volume and try again."
                        )
                        return
                else:
                    await progress.edit_text(
                        "RU:\n"
                        "Таймаут при обращении к Telegram Bot API (getFile).\n"
                        "Проверьте сервер и попробуйте ещё раз.\n\n"
                        "EN:\n"
                        "Timeout while calling Telegram Bot API (getFile).\n"
                        "Check server and try again."
                    )
                    return

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / f"input{file_ext}"
            if direct_source is not None:
                await progress.edit_text("Copying audio from local Bot API storage...")
                shutil.copyfile(direct_source, input_path)
                logger.info("Copied input via local storage source: %s -> %s", direct_source, input_path)
            elif tg_file is not None:
                try:
                    await tg_file.download_to_drive(custom_path=str(input_path))
                except (InvalidToken, BadRequest) as exc:
                    message_lower = str(exc).lower()
                    fallback_source = None
                    if bot_is_local_mode and ("not found" in message_lower or "invalid token" in message_lower):
                        fallback_source = _resolve_local_bot_api_file_path(tg_file.file_path)
                        if fallback_source is None:
                            fallback_source = await _wait_for_recent_local_media_file(
                                file_ext=file_ext,
                                expected_size=file_size,
                                timeout_seconds=_local_direct_pickup_wait_seconds(),
                            )
                    if fallback_source is not None:
                        shutil.copyfile(fallback_source, input_path)
                        logger.info(
                            "Downloaded file via local storage fallback: %s -> %s",
                            fallback_source,
                            input_path,
                        )
                    else:
                        if bot_is_local_mode and ("not found" in message_lower or "invalid token" in message_lower):
                            logger.warning(
                                "Local file endpoint failed and storage fallback could not resolve path. file_path=%r",
                                tg_file.file_path,
                            )
                            await progress.edit_text(
                                "RU:\n"
                                "Не удалось скачать файл через локальный file endpoint.\n"
                                "Для Docker укажите путь к data volume в TELEGRAM_LOCAL_DOCKER_DATA_DIR.\n"
                                "Пример: C:\\Users\\Gena\\Documents\\Playground\\tg-bot-api-data\n\n"
                                "EN:\n"
                                "Could not download file via local file endpoint.\n"
                                "For Docker, set TELEGRAM_LOCAL_DOCKER_DATA_DIR to your host data volume path.\n"
                                "Example: C:\\Users\\Gena\\Documents\\Playground\\tg-bot-api-data"
                            )
                            return
                        raise
            else:
                raise RuntimeError("No source available to download input audio file.")
            transcribe_input_path = input_path
            should_extract_audio = media_kind == "video" or file_ext.lower() in SUPPORTED_VIDEO_EXTENSIONS
            if should_extract_audio:
                with state_lock:
                    progress_state.update(
                        {
                            "stage": "finalizing",
                            "message": "Extracting audio track from video...",
                        }
                    )
                await progress.edit_text("Extracting audio track from video...")
                extracted_audio_path = tmp_path / "input_extracted.wav"
                await asyncio.to_thread(
                    _extract_audio_track_from_video,
                    input_path,
                    extracted_audio_path,
                )
                transcribe_input_path = extracted_audio_path
                logger.info(
                    "Extracted audio track from video: %s -> %s",
                    input_path,
                    extracted_audio_path,
                )

            with state_lock:
                progress_state.update(
                    {
                        "stage": "model_loading" if not transcriber.is_model_loaded() else "transcribing",
                        "message": (
                            "Preparing model for first run..."
                            if not transcriber.is_model_loaded()
                            else "Transcription started..."
                        ),
                        "done_chunks": 0,
                        "total_chunks": 0,
                    }
                )
            await progress.edit_text(
                f"Processing... (quality={quality}, language={language}, format={output_format})"
            )
            updater_task = asyncio.create_task(
                _progress_message_updater(progress, progress_state, state_lock, stop_event)
            )

            def _progress_callback(payload: dict[str, object]) -> None:
                with state_lock:
                    progress_state.update(payload)

            result = await asyncio.to_thread(
                transcriber.transcribe_file,
                transcribe_input_path,
                language,
                quality,
                output_format,
                speaker_mode,
                _progress_callback,
            )
            stop_event.set()
            if updater_task is not None:
                await updater_task
            await _finalize_and_send_transcription_result(
                message=message,
                progress=progress,
                context=context,
                user_id=user.id,
                transcriber=transcriber,
                result=result,
                quality=quality,
                output_format=output_format,
                speaker_mode=speaker_mode,
                post_model=post_model,
                state_lock=state_lock,
                progress_state=progress_state,
            )
    except Exception as exc:  # pylint: disable=broad-except
        stop_event.set()
        if updater_task is not None:
            await updater_task
        logger.exception("Failed to process audio")
        await progress.edit_text(f"Error: {exc}")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if err is None:
        return
    if isinstance(err, BadRequest) and "message is not modified" in str(err).lower():
        logger.debug("Ignoring harmless Telegram edit race: %s", err)
        return
    if isinstance(err, (TimedOut, NetworkError)):
        logger.warning("Transient Telegram network error: %s", err)
        return
    logger.exception("Unhandled error while processing update: %s", err)


def main() -> None:
    env_file = Path(__file__).with_name(".env")
    # Override process env with .env values to avoid empty shell vars shadowing token.
    load_dotenv(dotenv_path=env_file, override=True)
    _configure_cuda_paths()
    _configure_ffmpeg_paths()
    _start_local_bot_api_if_needed()
    atexit.register(_stop_local_bot_api_process)

    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip().strip('"').strip("'")
    if not token:
        token = _read_env_value_from_file("TELEGRAM_BOT_TOKEN", env_file)
    if not token:
        raise RuntimeError(f"Set TELEGRAM_BOT_TOKEN in environment or {env_file}.")

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
    diarization_backend = os.getenv("DIARIZATION_BACKEND", "nemo")
    nemo_num_speakers = int(os.getenv("NEMO_NUM_SPEAKERS", "0"))
    telegram_local_mode = _env_bool("TELEGRAM_LOCAL_MODE", False)
    telegram_api_base_url = os.getenv("TELEGRAM_API_BASE_URL", "").strip()
    telegram_api_file_url = os.getenv("TELEGRAM_API_FILE_URL", "").strip()
    telegram_connect_timeout = _env_float("TELEGRAM_CONNECT_TIMEOUT", 20.0 if telegram_local_mode else 5.0)
    telegram_read_timeout = _env_float("TELEGRAM_READ_TIMEOUT", 120.0 if telegram_local_mode else 5.0)
    telegram_write_timeout = _env_float("TELEGRAM_WRITE_TIMEOUT", 120.0 if telegram_local_mode else 5.0)
    telegram_media_write_timeout = _env_float(
        "TELEGRAM_MEDIA_WRITE_TIMEOUT", 180.0 if telegram_local_mode else 20.0
    )
    telegram_pool_timeout = _env_float("TELEGRAM_POOL_TIMEOUT", 15.0 if telegram_local_mode else 1.0)
    telegram_get_updates_read_timeout = _env_float(
        "TELEGRAM_GET_UPDATES_READ_TIMEOUT", 30.0 if telegram_local_mode else 5.0
    )

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
        diarization_backend=diarization_backend,
        nemo_num_speakers=nemo_num_speakers,
    )
    postprocessor = TextPostProcessor()
    runtime = transcriber.status()
    if runtime.get("nemo_available") is False:
        logger.warning("NeMo backend is unavailable: %s", runtime.get("nemo_reason"))
    if postprocessor.enabled:
        default_post_model = _default_postprocess_model()
        llm_status = postprocessor.runtime_status()
        logger.info(
            "Text post-processing is enabled (default model=%s, provider=%s, base=%s, prompts=%s).",
            _postprocess_model_label(default_post_model),
            postprocessor.provider,
            postprocessor.base_url,
            postprocessor.prompts_file,
        )
        if default_post_model == "oos20" and not llm_status.get("available"):
            logger.warning(
                "LM Studio preflight failed. Fallback mode will be used. Reason: %s",
                llm_status.get("error", "unknown"),
            )
    logger.info("Bot is starting.")

    app_builder = (
        Application.builder()
        .token(token)
        .connect_timeout(telegram_connect_timeout)
        .read_timeout(telegram_read_timeout)
        .write_timeout(telegram_write_timeout)
        .media_write_timeout(telegram_media_write_timeout)
        .pool_timeout(telegram_pool_timeout)
        .get_updates_read_timeout(telegram_get_updates_read_timeout)
    )
    if telegram_local_mode:
        if not telegram_api_base_url:
            telegram_api_base_url = _local_api_base_url()
        if not telegram_api_file_url:
            telegram_api_file_url = _local_api_base_url()
        telegram_api_base_url = _prefer_loopback_ipv4(telegram_api_base_url)
        telegram_api_file_url = _prefer_loopback_ipv4(telegram_api_file_url)

    normalized_bot_base_url = _normalize_bot_base_url(telegram_api_base_url) if telegram_api_base_url else ""
    normalized_file_base_url = _normalize_file_base_url(telegram_api_file_url) if telegram_api_file_url else ""
    if telegram_api_base_url:
        app_builder = app_builder.base_url(normalized_bot_base_url)
    if telegram_api_file_url:
        app_builder = app_builder.base_file_url(normalized_file_base_url)
    if telegram_local_mode:
        healthy, reason = _check_local_bot_api_get_me(token, normalized_bot_base_url or _local_api_base_url())
        if not healthy:
            raise RuntimeError(
                "Local Telegram Bot API preflight failed. "
                f"Reason: {reason}. Check Docker/telegram-bot-api server and TELEGRAM_API_BASE_URL."
            )
        app_builder = app_builder.local_mode(True)
        logger.info(
            "Telegram local Bot API mode is enabled. Base URL: %s",
            normalized_bot_base_url or _normalize_bot_base_url(_local_api_base_url()),
        )
    app = app_builder.build()
    app.bot_data["transcriber"] = transcriber
    app.bot_data["postprocessor"] = postprocessor
    app.add_error_handler(on_error)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("lang", set_language))
    app.add_handler(CommandHandler("quality", set_quality))
    app.add_handler(CommandHandler("format", set_format))
    app.add_handler(CommandHandler("model", set_postprocess_model))
    app.add_handler(CommandHandler("diar", set_diarization_backend))
    app.add_handler(CommandHandler("speakers", set_speakers))
    app.add_handler(CommandHandler("gpu", gpu_status_command))
    app.add_handler(CommandHandler("llm", llm_status_command))
    app.add_handler(
        CallbackQueryHandler(
            settings_callback,
            pattern=r"^(lang:|quality:|format:|model:|diar:|spk:|settings:)",
        )
    )
    app.add_handler(
        MessageHandler(
            filters.AUDIO | filters.VOICE | filters.VIDEO | filters.Document.AUDIO | filters.Document.VIDEO,
            transcribe_audio,
        )
    )
    app.add_handler(
        MessageHandler(
            filters.Document.ALL & (~filters.Document.AUDIO) & (~filters.Document.VIDEO),
            debug_text_document,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_instructions))

    # Python 3.14 no longer provides an implicit default event loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        _stop_local_bot_api_process()


if __name__ == "__main__":
    main()


