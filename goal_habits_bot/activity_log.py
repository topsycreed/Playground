from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _log_path() -> Path:
    raw = os.getenv("ACTIVITY_LOG_PATH", "data/activity_log.jsonl").strip()
    return Path(raw or "data/activity_log.jsonl")


def add_activity_entry(user_id: int, burned_kcal: float, source: str, note: str = "") -> None:
    value = float(burned_kcal)
    if value <= 0:
        raise ValueError("burned_kcal must be > 0")
    if value > 10000:
        raise ValueError("burned_kcal too large")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": int(user_id),
        "day": date.today().isoformat(),
        "burned_kcal": round(value, 2),
        "source": (source or "unknown").strip()[:32],
        "note": (note or "").strip()[:200],
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_line(line: str) -> dict[str, Any] | None:
    text = (line or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def activity_burned_today(user_id: int) -> float:
    path = _log_path()
    if not path.exists():
        return 0.0
    today = date.today().isoformat()
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = _parse_line(line)
        if not payload:
            continue
        if int(payload.get("user_id", -1)) != int(user_id):
            continue
        if str(payload.get("day") or "") != today:
            continue
        try:
            value = float(payload.get("burned_kcal", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            total += value
    return total
