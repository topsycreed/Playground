from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path


_LOCK = threading.Lock()
_VALID_SLOTS = {"breakfast", "lunch", "dinner"}


def _storage_path() -> Path:
    raw = os.getenv("MEAL_SKIP_PATH", "data/meal_skip.json").strip()
    return Path(raw or "data/meal_skip.json")


def _load() -> dict[str, dict[str, list[str]]]:
    path = _storage_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    out: dict[str, dict[str, list[str]]] = {}
    for user_key, by_day in payload.items():
        if not isinstance(user_key, str) or not isinstance(by_day, dict):
            continue
        clean_days: dict[str, list[str]] = {}
        for day_key, values in by_day.items():
            if not isinstance(day_key, str) or not isinstance(values, list):
                continue
            slots: list[str] = []
            for slot in values:
                name = str(slot).strip().lower()
                if name in _VALID_SLOTS and name not in slots:
                    slots.append(name)
            if slots:
                clean_days[day_key] = slots
        if clean_days:
            out[user_key] = clean_days
    return out


def _save(data: dict[str, dict[str, list[str]]]) -> None:
    path = _storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def skipped_slots_for_day(user_id: int, day: date | None = None) -> set[str]:
    target = (day or date.today()).isoformat()
    with _LOCK:
        data = _load()
        slots = data.get(str(int(user_id)), {}).get(target, [])
        return set(slots)


def skip_meal_slot(user_id: int, slot: str, day: date | None = None) -> bool:
    name = str(slot or "").strip().lower()
    if name not in _VALID_SLOTS:
        raise ValueError(f"Unsupported slot: {slot}")
    target = (day or date.today()).isoformat()
    user_key = str(int(user_id))
    with _LOCK:
        data = _load()
        by_day = data.setdefault(user_key, {})
        slots = by_day.setdefault(target, [])
        if name in slots:
            return False
        slots.append(name)
        _save(data)
        return True


def unskip_meal_slot(user_id: int, slot: str, day: date | None = None) -> bool:
    name = str(slot or "").strip().lower()
    if name not in _VALID_SLOTS:
        raise ValueError(f"Unsupported slot: {slot}")
    target = (day or date.today()).isoformat()
    user_key = str(int(user_id))
    with _LOCK:
        data = _load()
        by_day = data.get(user_key)
        if not by_day or target not in by_day:
            return False
        slots = [value for value in by_day.get(target, []) if value != name]
        if len(slots) == len(by_day.get(target, [])):
            return False
        if slots:
            by_day[target] = slots
        else:
            by_day.pop(target, None)
            if not by_day:
                data.pop(user_key, None)
        _save(data)
        return True
