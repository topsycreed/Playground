from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from nutrition_ai import MacroEstimate

_LOCK = threading.Lock()
MEAL_TYPE_MEAL = "meal"
MEAL_TYPE_SNACK = "snack"


def _log_path() -> Path:
    raw = os.getenv("MEAL_LOG_PATH", "data/meal_log.jsonl").strip() or "data/meal_log.jsonl"
    return Path(raw)


@dataclass
class DailySummary:
    meals_count: int
    total_kcal: float
    total_protein_g: float
    total_fat_g: float
    total_carbs_g: float
    avg_margin_percent: float


@dataclass
class DayStats:
    day: date
    meals_count: int
    total_kcal: float
    total_protein_g: float
    total_fat_g: float
    total_carbs_g: float
    avg_margin_percent: float


@dataclass
class MealEntry:
    timestamp: datetime
    meal_type: str
    dish_name: str
    calories_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float


def _normalize_meal_type(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token == MEAL_TYPE_SNACK:
        return MEAL_TYPE_SNACK
    return MEAL_TYPE_MEAL


def _is_primary_meal_row(row: dict[str, Any]) -> bool:
    return _normalize_meal_type(row.get("meal_type")) == MEAL_TYPE_MEAL


def append_meal(user_id: int, chat_id: int, estimate: MacroEstimate, meal_type: str = MEAL_TYPE_MEAL) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": int(user_id),
        "chat_id": int(chat_id),
        "meal_type": _normalize_meal_type(meal_type),
        "dish_name": estimate.dish_name,
        "portion_grams": estimate.portion_grams,
        "calories_kcal": estimate.calories_kcal,
        "protein_g": estimate.protein_g,
        "fat_g": estimate.fat_g,
        "carbs_g": estimate.carbs_g,
        "error_margin_percent": estimate.error_margin_percent,
        "confidence": estimate.confidence,
    }
    with _LOCK:
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


def _all_user_rows(user_id: int) -> list[dict[str, Any]]:
    path = _log_path()
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = _parse_line(line)
        if not payload:
            continue
        if int(payload.get("user_id", -1)) != int(user_id):
            continue
        ts = str(payload.get("timestamp") or "").strip()
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        payload["_dt"] = dt
        rows.append(payload)
    return rows


def _write_lines(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _summary_from_rows(rows: list[dict[str, Any]]) -> DailySummary:
    if not rows:
        return DailySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_kcal = sum(float(row.get("calories_kcal", 0.0) or 0.0) for row in rows)
    total_protein_g = sum(float(row.get("protein_g", 0.0) or 0.0) for row in rows)
    total_fat_g = sum(float(row.get("fat_g", 0.0) or 0.0) for row in rows)
    total_carbs_g = sum(float(row.get("carbs_g", 0.0) or 0.0) for row in rows)
    avg_margin_percent = (
        sum(float(row.get("error_margin_percent", 0.0) or 0.0) for row in rows) / len(rows)
    )
    return DailySummary(
        meals_count=len(rows),
        total_kcal=total_kcal,
        total_protein_g=total_protein_g,
        total_fat_g=total_fat_g,
        total_carbs_g=total_carbs_g,
        avg_margin_percent=avg_margin_percent,
    )


def summary_for_today(user_id: int) -> DailySummary:
    today = datetime.now().date()
    rows = [row for row in _all_user_rows(user_id) if row["_dt"].date() == today]
    return _summary_from_rows(rows)


def summary_all_time(user_id: int) -> DailySummary:
    return _summary_from_rows(_all_user_rows(user_id))


def summary_for_last_days(user_id: int, days: int) -> DailySummary:
    period_days = max(1, min(int(days), 3650))
    cutoff = datetime.now().date() - timedelta(days=period_days - 1)
    rows = [row for row in _all_user_rows(user_id) if row["_dt"].date() >= cutoff]
    return _summary_from_rows(rows)


def main_meals_count_for_day(user_id: int, day: date | None = None) -> int:
    target_day = day or datetime.now().date()
    return sum(
        1
        for row in _all_user_rows(user_id)
        if row["_dt"].date() == target_day and _is_primary_meal_row(row)
    )


def daily_history(user_id: int, days: int = 14) -> list[DayStats]:
    period_days = max(1, min(int(days), 3650))
    cutoff = datetime.now().date() - timedelta(days=period_days - 1)
    grouped: dict[date, list[dict[str, Any]]] = {}

    for row in _all_user_rows(user_id):
        day = row["_dt"].date()
        if day < cutoff:
            continue
        grouped.setdefault(day, []).append(row)

    result: list[DayStats] = []
    for day, rows in grouped.items():
        summary = _summary_from_rows(rows)
        result.append(
            DayStats(
                day=day,
                meals_count=summary.meals_count,
                total_kcal=summary.total_kcal,
                total_protein_g=summary.total_protein_g,
                total_fat_g=summary.total_fat_g,
                total_carbs_g=summary.total_carbs_g,
                avg_margin_percent=summary.avg_margin_percent,
            )
        )

    result.sort(key=lambda item: item.day, reverse=True)
    return result


def meals_count_in_window(
    user_id: int,
    *,
    start_hour: int,
    end_hour: int,
    day: date | None = None,
) -> int:
    start = max(0, min(int(start_hour), 23))
    end = max(start + 1, min(int(end_hour), 24))
    target_day = day or datetime.now().date()

    total = 0
    for row in _all_user_rows(user_id):
        dt = row["_dt"]
        if dt.date() != target_day:
            continue
        if not _is_primary_meal_row(row):
            continue
        if start <= dt.hour < end:
            total += 1
    return total


def meals_for_day(user_id: int, day: date | None = None) -> list[MealEntry]:
    target_day = day or datetime.now().date()
    rows = [row for row in _all_user_rows(user_id) if row["_dt"].date() == target_day]
    rows.sort(key=lambda row: row["_dt"])

    result: list[MealEntry] = []
    for row in rows:
        result.append(
            MealEntry(
                timestamp=row["_dt"],
                meal_type=_normalize_meal_type(row.get("meal_type")),
                dish_name=str(row.get("dish_name") or "Unknown dish"),
                calories_kcal=float(row.get("calories_kcal", 0.0) or 0.0),
                protein_g=float(row.get("protein_g", 0.0) or 0.0),
                fat_g=float(row.get("fat_g", 0.0) or 0.0),
                carbs_g=float(row.get("carbs_g", 0.0) or 0.0),
            )
        )
    return result


def delete_meal_for_day_index(user_id: int, day: date, index: int) -> MealEntry | None:
    item_index = int(index)
    if item_index <= 0:
        return None

    path = _log_path()
    if not path.exists():
        return None

    with _LOCK:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        matches: list[tuple[int, MealEntry]] = []

        for line_idx, line in enumerate(raw_lines):
            payload = _parse_line(line)
            if not payload:
                continue
            if int(payload.get("user_id", -1)) != int(user_id):
                continue
            ts = str(payload.get("timestamp") or "").strip()
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if dt.date() != day:
                continue
            matches.append(
                (
                    line_idx,
                    MealEntry(
                        timestamp=dt,
                        meal_type=_normalize_meal_type(payload.get("meal_type")),
                        dish_name=str(payload.get("dish_name") or "Unknown dish"),
                        calories_kcal=float(payload.get("calories_kcal", 0.0) or 0.0),
                        protein_g=float(payload.get("protein_g", 0.0) or 0.0),
                        fat_g=float(payload.get("fat_g", 0.0) or 0.0),
                        carbs_g=float(payload.get("carbs_g", 0.0) or 0.0),
                    ),
                )
            )

        matches.sort(key=lambda item: item[1].timestamp)
        if item_index > len(matches):
            return None

        target_line_idx, removed = matches[item_index - 1]
        raw_lines.pop(target_line_idx)
        _write_lines(path, raw_lines)
        return removed
