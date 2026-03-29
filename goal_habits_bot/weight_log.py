from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()


def _log_path() -> Path:
    raw = os.getenv("WEIGHT_LOG_PATH", "data/weight_log.jsonl").strip()
    return Path(raw or "data/weight_log.jsonl")


@dataclass
class WeightEntry:
    day: date
    weight_kg: float


@dataclass
class WeightLogEntry:
    timestamp: datetime
    day: date
    weight_kg: float


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


def add_weight(user_id: int, weight_kg: float, day: date | None = None) -> None:
    value = float(weight_kg)
    if value <= 0:
        raise ValueError("weight_kg must be > 0")
    if value > 500:
        raise ValueError("weight_kg too large")

    at_day = day or date.today()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": int(user_id),
        "day": at_day.isoformat(),
        "weight_kg": round(value, 2),
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _entries_for_user(user_id: int) -> list[WeightEntry]:
    path = _log_path()
    if not path.exists():
        return []
    out: list[WeightEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = _parse_line(line)
        if not payload:
            continue
        if int(payload.get("user_id", -1)) != int(user_id):
            continue
        day_raw = str(payload.get("day") or "").strip()
        try:
            day = date.fromisoformat(day_raw)
            weight = float(payload.get("weight_kg", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        out.append(WeightEntry(day=day, weight_kg=weight))
    out.sort(key=lambda item: item.day)
    return out


def latest_weight(user_id: int) -> WeightEntry | None:
    entries = _entries_for_user(user_id)
    if not entries:
        return None
    return entries[-1]


def first_weight(user_id: int) -> WeightEntry | None:
    entries = _entries_for_user(user_id)
    if not entries:
        return None
    return entries[0]


def weight_history(user_id: int) -> list[WeightEntry]:
    return _entries_for_user(user_id)


def weight_for_day(user_id: int, day: date | None = None) -> WeightEntry | None:
    target = day or date.today()
    entries = _entries_for_user(user_id)
    match: WeightEntry | None = None
    for item in entries:
        if item.day == target:
            match = item
    return match


def _write_lines(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def weights_for_day(user_id: int, day: date | None = None) -> list[WeightLogEntry]:
    target = day or date.today()
    path = _log_path()
    if not path.exists():
        return []

    out: list[WeightLogEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = _parse_line(line)
        if not payload:
            continue
        if int(payload.get("user_id", -1)) != int(user_id):
            continue
        day_raw = str(payload.get("day") or "").strip()
        ts_raw = str(payload.get("timestamp") or "").strip()
        try:
            row_day = date.fromisoformat(day_raw)
            ts = datetime.fromisoformat(ts_raw)
            weight = float(payload.get("weight_kg", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if row_day != target or weight <= 0:
            continue
        out.append(WeightLogEntry(timestamp=ts, day=row_day, weight_kg=weight))

    out.sort(key=lambda item: item.timestamp)
    return out


def delete_weight_for_day_index(user_id: int, day: date, index: int) -> WeightLogEntry | None:
    item_index = int(index)
    if item_index <= 0:
        return None

    path = _log_path()
    if not path.exists():
        return None

    with _LOCK:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        matches: list[tuple[int, WeightLogEntry]] = []
        for line_idx, line in enumerate(raw_lines):
            payload = _parse_line(line)
            if not payload:
                continue
            if int(payload.get("user_id", -1)) != int(user_id):
                continue
            day_raw = str(payload.get("day") or "").strip()
            ts_raw = str(payload.get("timestamp") or "").strip()
            try:
                row_day = date.fromisoformat(day_raw)
                ts = datetime.fromisoformat(ts_raw)
                weight = float(payload.get("weight_kg", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if row_day != day or weight <= 0:
                continue
            matches.append((line_idx, WeightLogEntry(timestamp=ts, day=row_day, weight_kg=weight)))

        matches.sort(key=lambda item: item[1].timestamp)
        if item_index > len(matches):
            return None

        target_line_idx, removed = matches[item_index - 1]
        raw_lines.pop(target_line_idx)
        _write_lines(path, raw_lines)
        return removed
