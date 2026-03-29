from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()


def _settings_path() -> Path:
    raw = os.getenv("USER_SETTINGS_PATH", "data/user_settings.json").strip()
    return Path(raw or "data/user_settings.json")


@dataclass
class UserSettings:
    birthdate: date | None
    weight_loss_goal_kg: float | None
    target_date: date | None


def _load_payload() -> dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _save_payload(payload: dict[str, Any]) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_birthdate(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def get_user_settings(user_id: int) -> UserSettings:
    key = str(int(user_id))
    with _LOCK:
        payload = _load_payload()
    row = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(row, dict):
        return UserSettings(birthdate=None, weight_loss_goal_kg=None, target_date=None)

    birthdate = _parse_birthdate(row.get("birthdate"))
    target_date = _parse_birthdate(row.get("target_date"))
    goal = row.get("weight_loss_goal_kg")
    try:
        goal_value = float(goal) if goal is not None else None
    except (TypeError, ValueError):
        goal_value = None
    if goal_value is not None and goal_value <= 0:
        goal_value = None
    return UserSettings(
        birthdate=birthdate,
        weight_loss_goal_kg=goal_value,
        target_date=target_date,
    )


def set_birthdate(user_id: int, birthdate: date) -> None:
    key = str(int(user_id))
    with _LOCK:
        payload = _load_payload()
        row = payload.get(key) if isinstance(payload.get(key), dict) else {}
        row["birthdate"] = birthdate.isoformat()
        row["updated_at"] = datetime.now().isoformat(timespec="seconds")
        payload[key] = row
        _save_payload(payload)


def set_weight_loss_goal(user_id: int, goal_kg: float) -> None:
    key = str(int(user_id))
    value = float(goal_kg)
    if value <= 0:
        raise ValueError("goal_kg must be > 0")
    with _LOCK:
        payload = _load_payload()
        row = payload.get(key) if isinstance(payload.get(key), dict) else {}
        row["weight_loss_goal_kg"] = round(value, 2)
        row["updated_at"] = datetime.now().isoformat(timespec="seconds")
        payload[key] = row
        _save_payload(payload)


def clear_weight_loss_goal(user_id: int) -> bool:
    key = str(int(user_id))
    with _LOCK:
        payload = _load_payload()
        row = payload.get(key)
        if not isinstance(row, dict) or "weight_loss_goal_kg" not in row:
            return False
        row.pop("weight_loss_goal_kg", None)
        row["updated_at"] = datetime.now().isoformat(timespec="seconds")
        payload[key] = row
        _save_payload(payload)
        return True


def set_target_date(user_id: int, value: date) -> None:
    key = str(int(user_id))
    with _LOCK:
        payload = _load_payload()
        row = payload.get(key) if isinstance(payload.get(key), dict) else {}
        row["target_date"] = value.isoformat()
        row["updated_at"] = datetime.now().isoformat(timespec="seconds")
        payload[key] = row
        _save_payload(payload)
