from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()


def _storage_path() -> Path:
    raw = os.getenv("REMINDER_SUBSCRIBERS_PATH", "data/reminder_subscribers.json").strip()
    return Path(raw or "data/reminder_subscribers.json")


def _load() -> list[int]:
    path = _storage_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []

    out: list[int] = []
    for item in payload:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value not in out:
            out.append(value)
    return out


def _save(chat_ids: list[int]) -> None:
    path = _storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chat_ids, ensure_ascii=False, indent=2), encoding="utf-8")


def list_subscribers() -> list[int]:
    with _LOCK:
        return _load()


def is_subscriber(chat_id: int) -> bool:
    with _LOCK:
        return int(chat_id) in _load()


def add_subscriber(chat_id: int) -> bool:
    chat_id = int(chat_id)
    with _LOCK:
        ids = _load()
        if chat_id in ids:
            return False
        ids.append(chat_id)
        _save(ids)
        return True


def remove_subscriber(chat_id: int) -> bool:
    chat_id = int(chat_id)
    with _LOCK:
        ids = _load()
        if chat_id not in ids:
            return False
        ids = [value for value in ids if value != chat_id]
        _save(ids)
        return True
