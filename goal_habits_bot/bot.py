from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from activity_ai import ActivityEstimate, GeminiActivityAnalyzer
from activity_log import activity_burned_today, add_activity_entry
from meal_skip_store import skip_meal_slot, skipped_slots_for_day, unskip_meal_slot
from meal_log import (
    MEAL_TYPE_MEAL,
    MEAL_TYPE_SNACK,
    append_meal,
    delete_meal_for_day_index,
    daily_history,
    main_meals_count_for_day,
    meals_for_day,
    meals_count_in_window,
    summary_all_time,
    summary_for_last_days,
    summary_for_today,
)
from advice_engine import AdviceResult, build_advice
from nutrition_ai import GeminiNutritionAnalyzer, GeminiRequestError, MacroEstimate, user_friendly_error
from reminder_store import add_subscriber, is_subscriber, list_subscribers, remove_subscriber
from user_settings import (
    clear_weight_loss_goal,
    get_user_settings,
    set_birthdate,
    set_target_date,
    set_weight_loss_goal,
)
from weight_log import add_weight, delete_weight_for_day_index, first_weight, latest_weight, weight_for_day, weights_for_day


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("goal_habits_bot")


def _guess_mime_type(file_path: Optional[str], fallback: str = "image/jpeg") -> str:
    if not file_path:
        return fallback
    value = file_path.lower()
    if value.endswith(".png"):
        return "image/png"
    if value.endswith(".webp"):
        return "image/webp"
    if value.endswith(".heic"):
        return "image/heic"
    return fallback


def _confidence_label(confidence: str) -> str:
    mapping = {
        "low": "Низкая",
        "medium": "Средняя",
        "high": "Высокая",
    }
    return mapping.get(confidence, "Средняя")


def _with_margin(value: float, margin_percent: float) -> tuple[float, float]:
    ratio = max(0.0, min(margin_percent, 100.0)) / 100.0
    low = max(0.0, value * (1.0 - ratio))
    high = max(0.0, value * (1.0 + ratio))
    return low, high


def _format_estimate(estimate: MacroEstimate) -> str:
    kcal_low, kcal_high = _with_margin(estimate.calories_kcal, estimate.error_margin_percent)
    p_low, p_high = _with_margin(estimate.protein_g, estimate.error_margin_percent)
    f_low, f_high = _with_margin(estimate.fat_g, estimate.error_margin_percent)
    c_low, c_high = _with_margin(estimate.carbs_g, estimate.error_margin_percent)

    lines = [
        f"Блюдо: {estimate.dish_name}",
        f"Порция: ~{estimate.portion_grams:.0f} г",
        "",
        f"Калории: ~{estimate.calories_kcal:.0f} ккал (диапазон {kcal_low:.0f}-{kcal_high:.0f})",
        f"Белки: ~{estimate.protein_g:.1f} г (диапазон {p_low:.1f}-{p_high:.1f})",
        f"Жиры: ~{estimate.fat_g:.1f} г (диапазон {f_low:.1f}-{f_high:.1f})",
        f"Углеводы: ~{estimate.carbs_g:.1f} г (диапазон {c_low:.1f}-{c_high:.1f})",
        "",
        f"Погрешность: ±{estimate.error_margin_percent:.0f}%",
        f"Уверенность оценки: {_confidence_label(estimate.confidence)}",
    ]
    if estimate.assumptions:
        lines.append("")
        lines.append("Предположения:")
        for assumption in estimate.assumptions[:3]:
            lines.append(f"- {assumption}")
    lines.append("")
    lines.append("Это оценка по фото, не медицинский расчёт.")
    return "\n".join(lines)


def _format_today_progress(meals_count: int, target: int) -> str:
    if target <= 0:
        return "Сегодня все приемы пищи помечены как пропущенные."
    left = max(0, target - meals_count)
    if left <= 0:
        return f"Сегодня: {meals_count}/{target} основных приемов. Цель по количеству выполнена."
    return f"Сегодня: {meals_count}/{target} основных приемов. Осталось: {left}."


def _format_today_summary(user_id: int) -> str:
    today = _today_local()
    target = _target_meals_for_day(user_id, today)
    skipped = _skipped_slots_for_day(user_id, today)
    main_meals_count = main_meals_count_for_day(user_id, today)
    today_items = meals_for_day(user_id, today)
    skipped_text = ""
    if skipped:
        labels = ", ".join(_slot_label(slot) for slot in _ordered_slots(skipped))
        skipped_text = f"\nПропущено: {labels}"

    summary = summary_for_today(user_id)
    burned_today = activity_burned_today(user_id)
    if summary.meals_count == 0:
        if burned_today > 0:
            return (
                f"Сегодня пока 0/{target} приемов пищи.\n"
                f"Активность: -{burned_today:.0f} ккал.\n"
                "Пришли фото первого блюда."
                + skipped_text
            )
        return f"Сегодня пока 0/{target} приемов пищи. Пришли фото первого блюда." + skipped_text

    today_items_text = ""
    if today_items:
        lines = ["", "Что ел сегодня:"]
        limit = 12
        for item in today_items[:limit]:
            kind = _meal_type_label(item.meal_type)
            lines.append(
                f"- {item.timestamp.strftime('%H:%M')} | {kind} | {item.dish_name} (~{item.calories_kcal:.0f} ккал)"
            )
        extra = len(today_items) - limit
        if extra > 0:
            lines.append(f"- ... еще {extra}")
        today_items_text = "\n".join(lines)

    kcal_low, kcal_high = _with_margin(summary.total_kcal, summary.avg_margin_percent)
    return (
        f"Сегодня: {main_meals_count}/{target} основных приемов\n"
        f"Калории: ~{summary.total_kcal:.0f} ккал (диапазон {kcal_low:.0f}-{kcal_high:.0f})\n"
        f"Активность: -{burned_today:.0f} ккал\n"
        f"Чистый баланс: ~{summary.total_kcal - burned_today:.0f} ккал\n"
        f"Белки: ~{summary.total_protein_g:.1f} г\n"
        f"Жиры: ~{summary.total_fat_g:.1f} г\n"
        f"Углеводы: ~{summary.total_carbs_g:.1f} г\n"
        f"Средняя погрешность: ±{summary.avg_margin_percent:.0f}%"
        + skipped_text
        + today_items_text
    )


def _format_history_summary(user_id: int, days: int) -> str:
    safe_days = max(1, min(int(days), 90))
    rows = daily_history(user_id, safe_days)
    if not rows:
        return f"За последние {safe_days} дн. данных нет. Пришли фото еды."

    lines = [f"История за {safe_days} дн.:"]
    for row in rows[:safe_days]:
        lines.append(
            f"{row.day.isoformat()}: {row.meals_count} приём., "
            f"{row.total_kcal:.0f} ккал, "
            f"Б/Ж/У {row.total_protein_g:.1f}/{row.total_fat_g:.1f}/{row.total_carbs_g:.1f}"
        )
    return "\n".join(lines)


def _format_stats_summary(user_id: int) -> str:
    all_time = summary_all_time(user_id)
    last7 = summary_for_last_days(user_id, 7)
    last30 = summary_for_last_days(user_id, 30)
    active_days = len(daily_history(user_id, 3650))
    activity_today = activity_burned_today(user_id)
    latest = latest_weight(user_id)
    progress = _build_weight_progress(user_id)

    lines = ["Статистика:"]
    if all_time.meals_count > 0:
        lines.extend(
            [
                (
                    f"За всё время: {all_time.meals_count} приёмов, {all_time.total_kcal:.0f} ккал, "
                    f"Б/Ж/У {all_time.total_protein_g:.1f}/{all_time.total_fat_g:.1f}/{all_time.total_carbs_g:.1f}"
                ),
                f"За 30 дней: {last30.meals_count} приёмов, {last30.total_kcal:.0f} ккал",
                f"За 7 дней: {last7.meals_count} приёмов, {last7.total_kcal:.0f} ккал",
                f"Активных дней с логами: {active_days}",
            ]
        )
    else:
        lines.append("По питанию пока мало данных. Добавь еду, и статистика начнет копиться.")

    lines.append(f"Активность сегодня: -{activity_today:.0f} ккал")
    if latest is not None:
        lines.append(f"Последний вес: {latest.weight_kg:.1f} кг ({latest.day.isoformat()})")

    if progress is not None:
        lines.extend(["", _format_weight_progress_block(progress), _weight_progress_feedback(progress)])
    else:
        lines.append("Для оценки прогресса по весу веди регулярные взвешивания и задай цель.")

    return "\n".join(lines)


def _calculate_age(birthdate: date, today: date | None = None) -> int:
    current = today or date.today()
    years = current.year - birthdate.year
    if (current.month, current.day) < (birthdate.month, birthdate.day):
        years -= 1
    return max(0, years)


@dataclass
class WeightProgress:
    start_day: date
    start_weight_kg: float
    current_day: date
    current_weight_kg: float
    goal_weight_kg: float
    goal_loss_kg: float
    actual_loss_kg: float
    expected_loss_kg: float
    delta_vs_plan_kg: float
    completion_percent: float
    elapsed_days: int
    total_days: int
    days_left: int
    status: str  # below | normal | above


def _build_weight_progress(
    user_id: int,
    *,
    today: date | None = None,
    today_weight_entry=None,
    settings=None,
) -> WeightProgress | None:
    current_day = today or _today_local()
    cfg = settings or get_user_settings(user_id)
    if cfg.weight_loss_goal_kg is None or cfg.target_date is None:
        return None

    start = first_weight(user_id)
    if start is None:
        return None

    current = today_weight_entry if today_weight_entry is not None else weight_for_day(user_id, current_day)
    if current is None:
        current = latest_weight(user_id)
    if current is None:
        return None

    total_days = (cfg.target_date - start.day).days
    if total_days <= 0:
        return None
    elapsed_days = (current_day - start.day).days
    elapsed_days = max(0, min(elapsed_days, total_days))
    days_left = max(0, (cfg.target_date - current_day).days)

    goal_loss = float(cfg.weight_loss_goal_kg)
    expected_loss = goal_loss * (elapsed_days / float(total_days))
    actual_loss = start.weight_kg - current.weight_kg
    delta = actual_loss - expected_loss
    completion = 0.0 if goal_loss <= 0 else (actual_loss / goal_loss) * 100.0
    completion = max(0.0, min(completion, 200.0))

    tolerance_kg = 0.7
    if delta > tolerance_kg:
        status = "above"
    elif delta < -tolerance_kg:
        status = "below"
    else:
        status = "normal"

    return WeightProgress(
        start_day=start.day,
        start_weight_kg=start.weight_kg,
        current_day=current.day,
        current_weight_kg=current.weight_kg,
        goal_weight_kg=start.weight_kg - goal_loss,
        goal_loss_kg=goal_loss,
        actual_loss_kg=actual_loss,
        expected_loss_kg=expected_loss,
        delta_vs_plan_kg=delta,
        completion_percent=completion,
        elapsed_days=elapsed_days,
        total_days=total_days,
        days_left=days_left,
        status=status,
    )


def _weight_progress_feedback(progress: WeightProgress) -> str:
    if progress.status == "above":
        return "Ты идешь чуть быстрее плана. Отличный темп, главное держать его комфортным и устойчивым."
    if progress.status == "below":
        return "Темп сейчас немного ниже плана, это нормально. Подкрутим режим мягко: сон, шаги и стабильный дефицит."
    return "Темп в норме относительно плана. Отличная стабильность, так держать."


def _format_weight_progress_block(progress: WeightProgress) -> str:
    status_map = {
        "below": "ниже плана",
        "normal": "в норме",
        "above": "выше плана",
    }
    status = status_map.get(progress.status, "в норме")
    return (
        "Прогресс по весу:\n"
        f"- Старт: {progress.start_weight_kg:.1f} кг ({progress.start_day.isoformat()})\n"
        f"- Текущий: {progress.current_weight_kg:.1f} кг ({progress.current_day.isoformat()})\n"
        f"- Целевой вес: ~{progress.goal_weight_kg:.1f} кг\n"
        f"- Снижение фактически: {progress.actual_loss_kg:.1f} кг из {progress.goal_loss_kg:.1f} кг ({progress.completion_percent:.0f}%)\n"
        f"- Ожидалось к сегодня: {progress.expected_loss_kg:.1f} кг\n"
        f"- Отклонение от плана: {progress.delta_vs_plan_kg:+.1f} кг ({status})\n"
        f"- До цели по времени: {progress.days_left} дн."
    )


def _format_profile(user_id: int) -> str:
    settings = get_user_settings(user_id)
    today_weight = weight_for_day(user_id)
    last_weight = latest_weight(user_id)
    activity_today = activity_burned_today(user_id)
    all_time = summary_all_time(user_id)

    birthdate_text = "не задана"
    age_text = "не задан"
    if settings.birthdate is not None:
        birthdate_text = settings.birthdate.isoformat()
        age_text = str(_calculate_age(settings.birthdate))

    goal_text = "не задана"
    if settings.weight_loss_goal_kg is not None:
        goal_text = f"-{settings.weight_loss_goal_kg:.1f} кг"

    target_date_text = settings.target_date.isoformat() if settings.target_date else "не задана"
    today_weight_text = "не задан"
    if today_weight is not None:
        today_weight_text = f"{today_weight.weight_kg:.1f} кг"
    elif last_weight is not None:
        today_weight_text = f"{last_weight.weight_kg:.1f} кг (последний от {last_weight.day.isoformat()})"

    return (
        "Профиль:\n"
        f"Дата рождения: {birthdate_text}\n"
        f"Возраст: {age_text}\n"
        f"Вес на сегодня: {today_weight_text}\n"
        f"Активность сегодня: -{activity_today:.0f} ккал\n"
        f"Цель по похудению: {goal_text}\n"
        f"Целевая дата: {target_date_text}\n"
        f"Всего приёмов пищи в логах: {all_time.meals_count}"
    )


def _format_advice(result: AdviceResult, progress: WeightProgress | None = None) -> str:
    lines = [
        "Совет на сегодня:",
        f"- Дней до цели: {result.days_left}",
        f"- Оценка поддержки: ~{result.maintenance_estimate:.0f} ккал/день",
        f"- Целевой дефицит: {result.deficit_used_per_day:.0f} ккал/день",
        "",
        "Цели на день:",
        f"- Калории: ~{result.daily_calorie_target:.0f} ккал",
        f"- Белки: ~{result.daily_protein_target:.1f} г",
        f"- Жиры: ~{result.daily_fat_target:.1f} г",
        f"- Углеводы: ~{result.daily_carbs_target:.1f} г",
        "",
        "Остаток на сегодня:",
        f"- Сожжено активностью: {result.burned_activity_kcal_today:.0f} ккал",
        f"- Чисто съедено: {result.net_consumed_kcal_today:.0f} ккал",
        f"- Калории: {result.remaining_calories:.0f} ккал",
        f"- Белки: {result.remaining_protein:.1f} г",
        f"- Жиры: {result.remaining_fat:.1f} г",
        f"- Углеводы: {result.remaining_carbs:.1f} г",
        "",
        result.meal_suggestion,
    ]
    if progress is not None:
        lines.extend(["", _format_weight_progress_block(progress), _weight_progress_feedback(progress)])
    if result.feasibility_warning:
        lines.extend(["", f"Важно: {result.feasibility_warning}"])
    return "\n".join(lines)


def _is_activity_caption(text: str) -> bool:
    payload = (text or "").strip().lower()
    return any(token in payload for token in ("#activity", "активност", "activity", "fitness", "fitbit"))


def _confidence_label_short(confidence: str) -> str:
    mapping = {"low": "низкая", "medium": "средняя", "high": "высокая"}
    return mapping.get(confidence, "средняя")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, GeminiRequestError):
        return int(exc.status_code or 0) in {408, 429, 500, 502, 503, 504}
    text = str(exc).lower()
    return any(token in text for token in ("timeout", "temporarily", "network", "rate limit", "quota"))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_reminder_times(raw: str, tz: tzinfo) -> list[dt_time]:
    defaults = [dt_time(9, 0, tzinfo=tz), dt_time(14, 0, tzinfo=tz), dt_time(20, 0, tzinfo=tz)]
    text = (raw or "").strip()
    if not text:
        return defaults

    cleaned = text.replace(";", ",")
    values: list[dt_time] = []
    seen: set[tuple[int, int]] = set()
    for token in cleaned.split(","):
        chunk = token.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 2:
            continue
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            continue
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            continue
        key = (hour, minute)
        if key in seen:
            continue
        seen.add(key)
        values.append(dt_time(hour, minute, tzinfo=tz))

    if not values:
        return defaults
    return sorted(values, key=lambda value: (value.hour, value.minute))


def _format_reminder_times(times: list[dt_time]) -> str:
    return ", ".join(value.strftime("%H:%M") for value in times)


def _today_local() -> date:
    return datetime.now(BOT_TZ).date()


def _ordered_slots(slots: set[str]) -> list[str]:
    return [slot for slot in SLOT_ORDER if slot in slots]


def _slot_label(slot: str) -> str:
    return SLOT_LABELS.get(str(slot), str(slot))


def _skipped_slots_for_day(user_id: int, day: date) -> set[str]:
    return skipped_slots_for_day(user_id, day)


def _target_meals_for_day(user_id: int, day: date) -> int:
    skipped = skipped_slots_for_day(user_id, day)
    planned = max(0, TARGET_MEALS_PER_DAY - len(skipped))
    return planned


def _parse_meal_slot_token(raw: str) -> str | None:
    value = (raw or "").strip().lower()
    if not value:
        return None
    mapping = {
        "breakfast": {
            "breakfast",
            "завтрак",
            "1",
            "утро",
            "утренний",
        },
        "lunch": {
            "lunch",
            "обед",
            "2",
            "день",
            "дневной",
        },
        "dinner": {
            "dinner",
            "ужин",
            "3",
            "вечер",
            "вечерний",
        },
    }
    for slot, aliases in mapping.items():
        if value in aliases:
            return slot
    return None


def _meal_feedback(estimate: MacroEstimate) -> str:
    issues: list[str] = []
    tips: list[str] = []

    if estimate.protein_g < 18:
        issues.append("белка маловато")
        tips.append("добавь источник белка (рыба, курица, яйца, творог, йогурт)")
    if estimate.calories_kcal > 900:
        issues.append("порция получилась довольно калорийной")
        tips.append("уменьши порцию или часть гарнира/соуса")
    if estimate.fat_g > 40:
        issues.append("жиров многовато")
        tips.append("снизь количество масла, жареного и жирных соусов")
    if estimate.carbs_g > 140:
        issues.append("углеводов многовато")
        tips.append("добавь больше овощей и белка вместо части углеводов")

    if not issues:
        return (
            "Отличный прием пищи, так держать. По оценке он выглядит достаточно сбалансированным "
            "и хорошо поддерживает цель по похудению."
        )

    unique_tips: list[str] = []
    for tip in tips:
        if tip not in unique_tips:
            unique_tips.append(tip)
    brief_issues = ", ".join(issues[:2])
    return (
        f"Неплохой прием пищи, но можно лучше: {brief_issues}. "
        "Чтобы идти к цели по похудению, попробуй: "
        + "; ".join(unique_tips[:2])
        + "."
    )


def _is_trivial_text_message(text: str) -> bool:
    payload = (text or "").strip().lower()
    if not payload:
        return True
    if len(payload) < 3:
        return True
    return payload in {
        "ок",
        "оке",
        "спасибо",
        "thanks",
        "понял",
        "поняла",
        "да",
        "нет",
        "привет",
        "hello",
        "hi",
    }


load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ANALYZER = GeminiNutritionAnalyzer.from_env()
ACTIVITY_ANALYZER = GeminiActivityAnalyzer.from_env()
TARGET_MEALS_PER_DAY = max(1, _env_int("TARGET_MEALS_PER_DAY", 3))

TIMEZONE_NAME = os.getenv("BOT_TIMEZONE", "Europe/Belgrade").strip() or "Europe/Belgrade"
try:
    BOT_TZ: tzinfo = ZoneInfo(TIMEZONE_NAME)
except ZoneInfoNotFoundError:
    logger.warning("Invalid BOT_TIMEZONE=%s; fallback to UTC.", TIMEZONE_NAME)
    TIMEZONE_NAME = "UTC"
    BOT_TZ = timezone.utc

REMINDER_TIMES = _parse_reminder_times(os.getenv("REMINDER_TIMES", "11:00,15:00,20:00"), BOT_TZ)
REMINDER_TIMES_LABEL = _format_reminder_times(REMINDER_TIMES)
REMINDER_TEXT = (
    os.getenv(
        "REMINDER_TEXT",
        "Напоминание о питании: что ты ел(а) в этот прием пищи? Пришли фото еды или короткое описание.",
    ).strip()
    or "Напоминание о питании: что ты ел(а) в этот прием пищи? Пришли фото еды или короткое описание."
)
JOB_QUEUE_READY = False
ACTIVITY_REMINDER_TIME = _parse_reminder_times(
    os.getenv("ACTIVITY_REMINDER_TIME", "21:30"),
    BOT_TZ,
)[0]
WEIGH_IN_REMINDER_TIME = _parse_reminder_times(
    os.getenv("WEIGH_IN_REMINDER_TIME", "07:00"),
    BOT_TZ,
)[0]
WEIGH_IN_REMINDER_TEXT = (
    os.getenv(
        "WEIGH_IN_REMINDER_TEXT",
        "Доброе утро! Взвесься до первого приема пищи и внеси вес: /set_weight N",
    ).strip()
    or "Доброе утро! Взвесься до первого приема пищи и внеси вес: /set_weight N"
)
ACTIVITY_REMINDER_TEXT = (
    os.getenv(
        "ACTIVITY_REMINDER_TEXT",
        "Вечерний чек: пришли скрин активности из фитнес-приложения (или /set_activity_kcal N). "
        "Сначала отправь /activity_photo.",
    ).strip()
    or "Вечерний чек: пришли скрин активности из фитнес-приложения (или /set_activity_kcal N). "
    "Сначала отправь /activity_photo."
)
MEAL_DEADLINE_SLOTS = [
    {
        "name": "breakfast",
        "label": "завтрак",
        "start_hour": 9,
        "end_hour": 11,
        "time": dt_time(11, 0, tzinfo=BOT_TZ),
        "check_weight": False,
    },
    {
        "name": "lunch",
        "label": "обед",
        "start_hour": 14,
        "end_hour": 15,
        "time": dt_time(15, 0, tzinfo=BOT_TZ),
        "check_weight": False,
    },
    {
        "name": "dinner",
        "label": "ужин",
        "start_hour": 15,
        "end_hour": 20,
        "time": dt_time(20, 0, tzinfo=BOT_TZ),
        "check_weight": False,
    },
]
MEAL_DEADLINES_LABEL = ", ".join(
    f"{item['label']} до {int(item['end_hour']):02d}:00" for item in MEAL_DEADLINE_SLOTS
)
SLOT_LABELS = {str(item["name"]): str(item["label"]) for item in MEAL_DEADLINE_SLOTS}
SLOT_ORDER = ["breakfast", "lunch", "dinner"]
PENDING_ACTIVITY_USERS: set[int] = set()
PENDING_INPUT_KEY = "pending_input"
NEXT_MEAL_TYPE_KEY = "next_meal_type"

MENU_BTN_TODAY = "Сегодня"
MENU_BTN_ADVICE = "Совет"
MENU_BTN_PROFILE = "Профиль"
MENU_BTN_MEAL = "Добавить еду"
MENU_BTN_SNACK = "Добавить перекус"
MENU_BTN_WEIGHT = "Вес сегодня"
MENU_BTN_GOAL = "Цель похудения"
MENU_BTN_TARGET_DATE = "Дата цели"
MENU_BTN_ACTIVITY_PHOTO = "Активность скрин"
MENU_BTN_ACTIVITY_KCAL = "Активность ккал"
MENU_BTN_HISTORY = "История"
MENU_BTN_STATS = "Статистика"
MENU_BTN_DELETE_MEAL = "Удалить еду"
MENU_BTN_DELETE_WEIGHT = "Удалить вес"
MENU_BTN_SKIP_MEAL = "Пропустить прием"
MENU_BTN_REMINDERS = "Напоминания"
MENU_BTN_REMINDERS_ON = "Напоминания Вкл"
MENU_BTN_REMINDERS_OFF = "Напоминания Выкл"
MENU_BTN_HELP = "Помощь"

PENDING_SET_WEIGHT = "set_weight"
PENDING_SET_GOAL = "set_goal"
PENDING_SET_TARGET_DATE = "set_target_date"
PENDING_SET_ACTIVITY_KCAL = "set_activity_kcal"
PENDING_SET_BIRTHDATE = "set_birthdate"
PENDING_DELETE_MEAL_TODAY = "delete_meal_today"
PENDING_DELETE_WEIGHT_TODAY = "delete_weight_today"
PENDING_SKIP_MEAL_TODAY = "skip_meal_today"


def _main_menu_markup() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [MENU_BTN_MEAL, MENU_BTN_SNACK, MENU_BTN_TODAY],
            [MENU_BTN_ADVICE],
            [MENU_BTN_WEIGHT, MENU_BTN_ACTIVITY_PHOTO, MENU_BTN_ACTIVITY_KCAL],
            [MENU_BTN_PROFILE, MENU_BTN_GOAL, MENU_BTN_TARGET_DATE],
            [MENU_BTN_HISTORY, MENU_BTN_STATS, MENU_BTN_SKIP_MEAL],
            [MENU_BTN_DELETE_MEAL, MENU_BTN_DELETE_WEIGHT],
            [MENU_BTN_REMINDERS, MENU_BTN_REMINDERS_ON, MENU_BTN_REMINDERS_OFF],
            [MENU_BTN_HELP],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        input_field_placeholder="Фото еды, /meal описание или кнопки меню",
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        "Привет. Я Goal/Habits bot.\n"
        "Отправляй фото еды или текст через /meal, и я оценю калории и БЖУ.\n"
        f"Цель на день: {TARGET_MEALS_PER_DAY} приема пищи.\n"
        f"Напоминания по еде: {REMINDER_TIMES_LABEL} ({TIMEZONE_NAME}).\n"
        f"Тихое утреннее взвешивание: {WEIGH_IN_REMINDER_TIME.strftime('%H:%M')} ({TIMEZONE_NAME}).\n"
        f"Вечерняя активность: {ACTIVITY_REMINDER_TIME.strftime('%H:%M')} ({TIMEZONE_NAME}).\n"
        "Нажимай кнопки меню ниже, чтобы не вводить команды вручную.",
        reply_markup=_main_menu_markup(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        "Как пользоваться ботом\n\n"
        "- /menu или /start: показать кнопки меню\n"
        "- Еда: фото или текст через /meal описание\n"
        "- Перекус: /snack описание или кнопка 'Добавить перекус'\n"
        "- После каждого приема: оценка + фидбек\n\n"
        "Пропуск приема пищи:\n"
        "- /skip_meal завтрак|обед|ужин\n"
        "- /unskip_meal завтрак|обед|ужин\n"
        "- /skips_today - что пропущено сегодня\n"
        "Если прием пропущен, бот не напоминает по нему и считает оставшиеся приемы.\n\n"
        "Удаление записей за сегодня:\n"
        "- /meals_today, /delete_meal_today N\n"
        "- /weights_today, /delete_weight_today N\n"
        "Перекусы удаляются там же, в списке приемов пищи.\n"
        "Если после удаления данных снова нет, напоминания вернутся.\n\n"
        "Окна питания:\n"
        "- завтрак: 09:00-11:00\n"
        "- обед: 14:00-15:00\n"
        "- ужин: 15:00-20:00\n"
        f"- тихое напоминание о весе: {WEIGH_IN_REMINDER_TIME.strftime('%H:%M')}"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(f"Текущая модель: {ANALYZER.model}")


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        "Меню включено. Можно пользоваться кнопками ниже.",
        reply_markup=_main_menu_markup(),
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    await message.reply_text(_format_today_summary(user.id))


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    await message.reply_text(_format_profile(user.id))


def _set_pending_input(context: ContextTypes.DEFAULT_TYPE, action: str | None) -> None:
    if action:
        context.user_data[PENDING_INPUT_KEY] = action
        return
    context.user_data.pop(PENDING_INPUT_KEY, None)


def _normalize_meal_type(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token == MEAL_TYPE_SNACK:
        return MEAL_TYPE_SNACK
    return MEAL_TYPE_MEAL


def _set_next_meal_type(context: ContextTypes.DEFAULT_TYPE, meal_type: str | None) -> None:
    if meal_type is None:
        context.user_data.pop(NEXT_MEAL_TYPE_KEY, None)
        return
    context.user_data[NEXT_MEAL_TYPE_KEY] = _normalize_meal_type(meal_type)


def _peek_next_meal_type(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    value = context.user_data.get(NEXT_MEAL_TYPE_KEY)
    if not value:
        return None
    return _normalize_meal_type(str(value))


def _consume_next_meal_type(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    value = _peek_next_meal_type(context)
    if value is not None:
        context.user_data.pop(NEXT_MEAL_TYPE_KEY, None)
    return value


def _meal_type_label(meal_type: str) -> str:
    return "перекус" if _normalize_meal_type(meal_type) == MEAL_TYPE_SNACK else "прием пищи"


def _is_snack_caption(text: str) -> bool:
    payload = (text or "").strip().lower()
    return any(token in payload for token in ("#snack", "перекус", "snack"))


async def _apply_birthdate_raw(message, user_id: int, raw: str) -> bool:
    value = (raw or "").strip()
    try:
        birthdate = date.fromisoformat(value)
    except ValueError:
        await message.reply_text("Неверный формат даты. Пример: 1992-08-14")
        return False

    if birthdate >= _today_local():
        await message.reply_text("Дата рождения должна быть в прошлом.")
        return False

    age = _calculate_age(birthdate)
    if age > 120:
        await message.reply_text("Проверь дату рождения: возраст получился слишком большим.")
        return False

    await asyncio.to_thread(set_birthdate, user_id, birthdate)
    await message.reply_text(f"Сохранил дату рождения: {birthdate.isoformat()} (возраст: {age}).")
    return True


async def _apply_goal_raw(message, user_id: int, raw: str) -> bool:
    value = (raw or "").replace(",", ".").strip()
    try:
        goal = float(value)
    except ValueError:
        await message.reply_text("Не понял число. Пример: 8.5")
        return False

    if goal <= 0 or goal > 200:
        await message.reply_text("Цель должна быть в диапазоне (0, 200] кг.")
        return False

    await asyncio.to_thread(set_weight_loss_goal, user_id, goal)
    await message.reply_text(f"Сохранил цель: сбросить {goal:.1f} кг.")
    return True


async def _apply_target_date_raw(message, user_id: int, raw: str) -> bool:
    value = (raw or "").strip()
    try:
        target = date.fromisoformat(value)
    except ValueError:
        await message.reply_text("Неверный формат. Пример: 2026-09-01")
        return False

    if target <= _today_local():
        await message.reply_text("Целевая дата должна быть в будущем.")
        return False

    await asyncio.to_thread(set_target_date, user_id, target)
    await message.reply_text(f"Сохранил целевую дату: {target.isoformat()}.")
    return True


async def _apply_weight_raw(message, user_id: int, raw: str) -> bool:
    value = (raw or "").replace(",", ".").strip()
    try:
        weight = float(value)
    except ValueError:
        await message.reply_text("Не понял число. Пример: 82.4")
        return False

    if weight <= 0 or weight > 500:
        await message.reply_text("Вес должен быть в диапазоне (0, 500] кг.")
        return False

    await asyncio.to_thread(add_weight, user_id, weight, _today_local())
    await message.reply_text(f"Сохранил вес на сегодня: {weight:.1f} кг.")
    return True


async def _apply_activity_kcal_raw(message, user_id: int, raw: str) -> bool:
    value = (raw or "").replace(",", ".").strip()
    try:
        kcal = float(value)
    except ValueError:
        await message.reply_text("Не понял число. Пример: 420")
        return False

    if kcal <= 0 or kcal > 10000:
        await message.reply_text("Значение должно быть в диапазоне (0, 10000] ккал.")
        return False

    await asyncio.to_thread(add_activity_entry, user_id, kcal, "manual", "manual command")
    burned = await asyncio.to_thread(activity_burned_today, user_id)
    await message.reply_text(
        f"Сохранил активность: {kcal:.0f} ккал.\nСуммарно сожжено сегодня: {burned:.0f} ккал."
    )
    return True


async def _format_meals_today_for_delete(user_id: int) -> str:
    items = await asyncio.to_thread(meals_for_day, user_id, _today_local())
    if not items:
        return "За сегодня приемов пищи нет."
    lines = ["Приемы пищи за сегодня:"]
    for idx, item in enumerate(items, start=1):
        kind = _meal_type_label(item.meal_type)
        lines.append(f"{idx}. {item.timestamp.strftime('%H:%M')} | {kind} | {item.dish_name} | ~{item.calories_kcal:.0f} ккал")
    lines.append("Отправь номер для удаления.")
    return "\n".join(lines)


async def _format_weights_today_for_delete(user_id: int) -> str:
    items = await asyncio.to_thread(weights_for_day, user_id, _today_local())
    if not items:
        return "За сегодня записей веса нет."
    lines = ["Записи веса за сегодня:"]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item.timestamp.strftime('%H:%M')} | {item.weight_kg:.1f} кг")
    lines.append("Отправь номер для удаления.")
    return "\n".join(lines)


async def _apply_delete_meal_today_raw(message, user_id: int, raw: str) -> bool:
    value = (raw or "").strip()
    try:
        index = int(value)
    except ValueError:
        await message.reply_text("Нужен номер из списка. Пример: 1")
        return False

    removed = await asyncio.to_thread(delete_meal_for_day_index, user_id, _today_local(), index)
    if removed is None:
        await message.reply_text("Не нашел запись с таким номером.")
        return False

    await message.reply_text(
        f"Удалил запись: {removed.timestamp.strftime('%H:%M')} | {_meal_type_label(removed.meal_type)} | {removed.dish_name} | ~{removed.calories_kcal:.0f} ккал."
    )
    return True


async def _apply_delete_weight_today_raw(message, user_id: int, raw: str) -> bool:
    value = (raw or "").strip()
    try:
        index = int(value)
    except ValueError:
        await message.reply_text("Нужен номер из списка. Пример: 1")
        return False

    removed = await asyncio.to_thread(delete_weight_for_day_index, user_id, _today_local(), index)
    if removed is None:
        await message.reply_text("Не нашел запись веса с таким номером.")
        return False

    await message.reply_text(
        f"Удалил запись веса: {removed.timestamp.strftime('%H:%M')} | {removed.weight_kg:.1f} кг."
    )
    return True


async def set_birthdate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if not context.args:
        _set_pending_input(context, PENDING_SET_BIRTHDATE)
        await message.reply_text("Пришли дату рождения в формате YYYY-MM-DD. Пример: 1992-08-14")
        return

    ok = await _apply_birthdate_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


async def set_goal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if not context.args:
        _set_pending_input(context, PENDING_SET_GOAL)
        await message.reply_text("Сколько кг хочешь сбросить? Пример: 8.5")
        return

    ok = await _apply_goal_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


async def set_target_date_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if not context.args:
        _set_pending_input(context, PENDING_SET_TARGET_DATE)
        await message.reply_text("Пришли целевую дату в формате YYYY-MM-DD. Пример: 2026-09-01")
        return

    ok = await _apply_target_date_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


async def set_weight_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if not context.args:
        _set_pending_input(context, PENDING_SET_WEIGHT)
        await message.reply_text("Пришли вес на сегодня в кг. Пример: 82.4")
        return

    ok = await _apply_weight_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


async def advice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    settings = await asyncio.to_thread(get_user_settings, user.id)
    today_weight = await asyncio.to_thread(weight_for_day, user.id, _today_local())

    missing: list[str] = []
    if settings.birthdate is None:
        missing.append("/set_birthdate YYYY-MM-DD")
    if today_weight is None:
        missing.append("/set_weight N")
    if settings.weight_loss_goal_kg is None:
        missing.append("/set_goal N")
    if settings.target_date is None:
        missing.append("/set_target_date YYYY-MM-DD")

    if missing:
        await message.reply_text(
            "Для совета не хватает данных. Заполни:\n- " + "\n- ".join(missing)
        )
        return

    age = _calculate_age(settings.birthdate)
    consumed = await asyncio.to_thread(summary_for_today, user.id)
    burned_today = await asyncio.to_thread(activity_burned_today, user.id)
    try:
        result = build_advice(
            age=age,
            weight_kg=today_weight.weight_kg,
            goal_loss_kg=settings.weight_loss_goal_kg,
            target_date=settings.target_date,
            today=_today_local(),
            consumed_today=consumed,
            burned_activity_kcal_today=burned_today,
        )
    except ValueError as exc:
        await message.reply_text(f"Не могу посчитать совет: {exc}")
        return

    progress = await asyncio.to_thread(
        _build_weight_progress,
        user.id,
        today=_today_local(),
        today_weight_entry=today_weight,
        settings=settings,
    )
    await message.reply_text(_format_advice(result, progress))


async def activity_photo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    PENDING_ACTIVITY_USERS.add(user.id)
    await message.reply_text(
        "Ок, жду скрин активности. Пришли изображение из фитнес-приложения (ккал за день)."
    )


async def set_activity_kcal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if not context.args:
        _set_pending_input(context, PENDING_SET_ACTIVITY_KCAL)
        await message.reply_text("Сколько ккал сжег за активность? Пример: 420")
        return

    ok = await _apply_activity_kcal_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


async def meals_today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    await message.reply_text(await _format_meals_today_for_delete(user.id), reply_markup=_main_menu_markup())


async def delete_meal_today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    if not context.args:
        _set_pending_input(context, PENDING_DELETE_MEAL_TODAY)
        await message.reply_text(await _format_meals_today_for_delete(user.id), reply_markup=_main_menu_markup())
        return

    ok = await _apply_delete_meal_today_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


async def weights_today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    await message.reply_text(await _format_weights_today_for_delete(user.id), reply_markup=_main_menu_markup())


async def delete_weight_today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    if not context.args:
        _set_pending_input(context, PENDING_DELETE_WEIGHT_TODAY)
        await message.reply_text(await _format_weights_today_for_delete(user.id), reply_markup=_main_menu_markup())
        return

    ok = await _apply_delete_weight_today_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


def _format_skips_today(user_id: int) -> str:
    skipped = _skipped_slots_for_day(user_id, _today_local())
    if not skipped:
        return "На сегодня пропусков нет."
    labels = ", ".join(_slot_label(slot) for slot in _ordered_slots(skipped))
    return f"Пропуски на сегодня: {labels}."


async def _apply_skip_meal_today_raw(message, user_id: int, raw: str) -> bool:
    slot = _parse_meal_slot_token(raw)
    if slot is None:
        await message.reply_text("Укажи прием: завтрак/обед/ужин (или 1/2/3).")
        return False

    added = await asyncio.to_thread(skip_meal_slot, user_id, slot, _today_local())
    if not added:
        await message.reply_text(f"{_slot_label(slot).capitalize()} уже отмечен как пропущенный.")
        return False

    await message.reply_text(
        f"Ок, отметил пропуск: {_slot_label(slot)}. Напоминания по этому приему сегодня отключены."
    )
    return True


async def _apply_unskip_meal_today_raw(message, user_id: int, raw: str) -> bool:
    slot = _parse_meal_slot_token(raw)
    if slot is None:
        await message.reply_text("Укажи прием: завтрак/обед/ужин (или 1/2/3).")
        return False

    removed = await asyncio.to_thread(unskip_meal_slot, user_id, slot, _today_local())
    if not removed:
        await message.reply_text(f"{_slot_label(slot).capitalize()} не был помечен как пропуск.")
        return False

    await message.reply_text(f"Вернул {_slot_label(slot)} в план на сегодня.")
    return True


async def skip_meal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    if not context.args:
        _set_pending_input(context, PENDING_SKIP_MEAL_TODAY)
        await message.reply_text(
            "Какой прием пропускаешь сегодня? Напиши: завтрак, обед или ужин (можно 1/2/3).",
            reply_markup=_main_menu_markup(),
        )
        return

    ok = await _apply_skip_meal_today_raw(message, user.id, context.args[0])
    if ok:
        _set_pending_input(context, None)


async def unskip_meal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    if not context.args:
        await message.reply_text("Используй: /unskip_meal завтрак|обед|ужин")
        return

    await _apply_unskip_meal_today_raw(message, user.id, context.args[0])


async def skips_today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    await message.reply_text(_format_skips_today(user.id), reply_markup=_main_menu_markup())


async def clear_goal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    removed = await asyncio.to_thread(clear_weight_loss_goal, user.id)
    if removed:
        await message.reply_text("Цель по похудению очищена.")
    else:
        await message.reply_text("Цель по похудению уже не была задана.")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    days = 14
    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            await message.reply_text("Используй: /history или /history 30")
            return

    await message.reply_text(_format_history_summary(user.id, days))


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    await message.reply_text(_format_stats_summary(user.id))


async def reminders_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return

    enabled = await asyncio.to_thread(is_subscriber, chat.id)
    status = "включены" if enabled else "выключены"
    scheduler_status = "активен" if JOB_QUEUE_READY else "не активен"
    await message.reply_text(
        f"Напоминания {status}.\n"
        f"Планировщик: {scheduler_status}\n"
        f"Время: {REMINDER_TIMES_LABEL} ({TIMEZONE_NAME})\n"
        f"Утреннее взвешивание: {WEIGH_IN_REMINDER_TIME.strftime('%H:%M')} ({TIMEZONE_NAME}), тихое\n"
        f"Текст: {REMINDER_TEXT}\n"
        "Команды: /reminders_on /reminders_off"
    )


async def reminders_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return

    added = await asyncio.to_thread(add_subscriber, chat.id)
    if added:
        await message.reply_text(
            f"Готово. Напоминания включены: {REMINDER_TIMES_LABEL} ({TIMEZONE_NAME})."
        )
    else:
        await message.reply_text("Напоминания уже включены.")


async def reminders_off_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return

    removed = await asyncio.to_thread(remove_subscriber, chat.id)
    if removed:
        await message.reply_text("Ок, напоминания выключены.")
    else:
        await message.reply_text("Напоминания уже были выключены.")


async def reminders_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await asyncio.to_thread(list_subscribers)
    if not chat_ids:
        return

    now_local = datetime.now(BOT_TZ)
    today = now_local.date()
    current_hour = now_local.hour

    for chat_id in chat_ids:
        text = REMINDER_TEXT
        try:
            user_id = int(chat_id) if int(chat_id) > 0 else None
            extra_lines: list[str] = []
            if user_id is not None:
                skipped_today = await asyncio.to_thread(skipped_slots_for_day, user_id, today)

                for slot in MEAL_DEADLINE_SLOTS:
                    if str(slot["name"]) in skipped_today:
                        continue
                    end_hour = int(slot["end_hour"])
                    if current_hour < end_hour:
                        continue
                    meals_in_slot = await asyncio.to_thread(
                        meals_count_in_window,
                        user_id,
                        start_hour=int(slot["start_hour"]),
                        end_hour=end_hour,
                        day=today,
                    )
                    if meals_in_slot <= 0:
                        extra_lines.append(
                            f"Нет записи за {slot['label']} (до {end_hour:02d}:00). Пришли фото еды или /meal."
                        )

            if extra_lines:
                text = text + "\n\nПроверь цели на сегодня:\n- " + "\n- ".join(extra_lines)

            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:
            logger.warning("Failed to send reminder to chat %s: %s", chat_id, exc)


async def morning_weigh_in_reminder_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await asyncio.to_thread(list_subscribers)
    if not chat_ids:
        return
    today = _today_local()
    for chat_id in chat_ids:
        try:
            user_id = int(chat_id) if int(chat_id) > 0 else None
            if user_id is None:
                continue
            today_weight = await asyncio.to_thread(weight_for_day, user_id, today)
            if today_weight is not None:
                continue
            await context.bot.send_message(
                chat_id=chat_id,
                text=WEIGH_IN_REMINDER_TEXT,
                disable_notification=True,
            )
        except Exception as exc:
            logger.warning("Failed to send weigh-in reminder to chat %s: %s", chat_id, exc)


async def activity_reminder_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = await asyncio.to_thread(list_subscribers)
    if not chat_ids:
        return
    for chat_id in chat_ids:
        try:
            await context.bot.send_message(chat_id=chat_id, text=ACTIVITY_REMINDER_TEXT)
            if int(chat_id) > 0:
                # In private chats, chat_id usually equals user_id.
                PENDING_ACTIVITY_USERS.add(int(chat_id))
        except Exception as exc:
            logger.warning("Failed to send activity reminder to chat %s: %s", chat_id, exc)


async def meal_deadline_check_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    payload = context.job.data if context.job else {}
    slot_name = str((payload or {}).get("name") or "")
    slot_label = str((payload or {}).get("label") or "прием пищи")
    start_hour = int((payload or {}).get("start_hour") or 0)
    end_hour = int((payload or {}).get("end_hour") or 24)
    check_weight = bool((payload or {}).get("check_weight"))
    today = _today_local()

    chat_ids = await asyncio.to_thread(list_subscribers)
    if not chat_ids:
        return

    for chat_id in chat_ids:
        reminders: list[str] = []
        user_id = int(chat_id) if int(chat_id) > 0 else None
        if user_id is not None:
            skipped_today = await asyncio.to_thread(skipped_slots_for_day, user_id, today)
            is_slot_skipped = bool(slot_name and slot_name in skipped_today)
            if not is_slot_skipped:
                meals_in_slot = await asyncio.to_thread(
                    meals_count_in_window,
                    user_id,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    day=today,
                )
                if meals_in_slot <= 0:
                    reminders.append(
                        f"до {end_hour:02d}:00 не вижу записи за {slot_label} (фото или /meal описание)."
                    )
            if check_weight:
                today_weight = await asyncio.to_thread(weight_for_day, user_id, today)
                if today_weight is None:
                    reminders.append("взвешивание за сегодня не зафиксировано (/set_weight N).")
        else:
            reminders.append(
                f"если {slot_label} еще не зафиксирован, отправь фото еды или текст через /meal."
            )

        if not reminders:
            continue

        text = "Напоминание по целям:\n- " + "\n- ".join(reminders)
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:
            logger.warning("Failed to send meal deadline reminder to chat %s: %s", chat_id, exc)


def schedule_reminders(app) -> None:
    global JOB_QUEUE_READY
    if app.job_queue is None:
        JOB_QUEUE_READY = False
        logger.warning("JobQueue unavailable. Install python-telegram-bot[job-queue].")
        return

    for idx, reminder_time in enumerate(REMINDER_TIMES, start=1):
        app.job_queue.run_daily(
            reminders_job,
            time=reminder_time,
            days=(0, 1, 2, 3, 4, 5, 6),
            name=f"meal-reminder-{idx}",
        )
    app.job_queue.run_daily(
        morning_weigh_in_reminder_job,
        time=WEIGH_IN_REMINDER_TIME,
        days=(0, 1, 2, 3, 4, 5, 6),
        name="weigh-in-reminder",
    )

    app.job_queue.run_daily(
        activity_reminder_job,
        time=ACTIVITY_REMINDER_TIME,
        days=(0, 1, 2, 3, 4, 5, 6),
        name="activity-reminder",
    )
    for slot in MEAL_DEADLINE_SLOTS:
        app.job_queue.run_daily(
            meal_deadline_check_job,
            time=slot["time"],
            days=(0, 1, 2, 3, 4, 5, 6),
            name=f"meal-deadline-{slot['name']}",
            data=slot,
        )
    JOB_QUEUE_READY = True
    logger.info(
        "Scheduled reminder jobs: meals=%s at %s, weigh-in=%s, activity=%s, deadlines=%s (%s)",
        len(REMINDER_TIMES),
        REMINDER_TIMES_LABEL,
        WEIGH_IN_REMINDER_TIME.strftime("%H:%M"),
        ACTIVITY_REMINDER_TIME.strftime("%H:%M"),
        ",".join(item["time"].strftime("%H:%M") for item in MEAL_DEADLINE_SLOTS),
        TIMEZONE_NAME,
    )


async def _analyze_activity_photo(
    *,
    user_id: int,
    message,
    bot,
    file_id: str,
    mime_type: str,
) -> None:
    progress = await message.reply_text("Читаю скрин активности...")
    try:
        telegram_file = await bot.get_file(file_id)
        image_data = await telegram_file.download_as_bytearray()
        if message.photo:
            mime_type = _guess_mime_type(telegram_file.file_path, fallback="image/jpeg")

        estimate: ActivityEstimate = await asyncio.to_thread(
            ACTIVITY_ANALYZER.extract_activity,
            bytes(image_data),
            mime_type,
        )
        if estimate.burned_kcal <= 0:
            await progress.edit_text(
                "Не увидел значение ккал на скрине. Можешь ввести вручную: /set_activity_kcal N"
            )
            return

        note = estimate.source_text if estimate.source_text else "screenshot"
        await asyncio.to_thread(add_activity_entry, user_id, estimate.burned_kcal, "screenshot", note)
        burned = await asyncio.to_thread(activity_burned_today, user_id)
        await progress.edit_text(
            "Активность сохранена:\n"
            f"- Добавлено: {estimate.burned_kcal:.0f} ккал\n"
            f"- Уверенность: {_confidence_label_short(estimate.confidence)}\n"
            f"- Итого сожжено сегодня: {burned:.0f} ккал"
        )
    except Exception as exc:
        logger.exception("Failed to analyze activity screenshot: %s", exc)
        await progress.edit_text(
            "Не получилось распознать активность на скрине.\n"
            f"Причина: {user_friendly_error(exc)}\n"
            "Можно ввести вручную: /set_activity_kcal N"
        )


async def analyze_food_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    bot = update.get_bot()
    user = update.effective_user
    chat = update.effective_chat

    file_id: Optional[str] = None
    mime_type = "image/jpeg"
    if message.photo:
        largest_photo = message.photo[-1]
        file_id = largest_photo.file_id
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        file_id = message.document.file_id
        mime_type = message.document.mime_type or "image/jpeg"
    else:
        await message.reply_text("Пришли фото еды как изображение.")
        return

    caption = str(message.caption or "")
    if user is not None:
        activity_mode = False
        if user.id in PENDING_ACTIVITY_USERS:
            PENDING_ACTIVITY_USERS.discard(user.id)
            activity_mode = True
        elif _is_activity_caption(caption):
            activity_mode = True
        if activity_mode:
            await _analyze_activity_photo(
                user_id=user.id,
                message=message,
                bot=bot,
                file_id=file_id,
                mime_type=mime_type,
            )
            return

    selected_meal_type = MEAL_TYPE_MEAL
    if _is_snack_caption(caption):
        selected_meal_type = MEAL_TYPE_SNACK
        _set_next_meal_type(context, None)
    else:
        selected_meal_type = _consume_next_meal_type(context) or MEAL_TYPE_MEAL

    progress = await message.reply_text("Анализирую фото...")
    try:
        telegram_file = await bot.get_file(file_id)
        image_data = await telegram_file.download_as_bytearray()
        if message.photo:
            mime_type = _guess_mime_type(telegram_file.file_path, fallback="image/jpeg")

        estimate: MacroEstimate | None = None
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                estimate = await asyncio.to_thread(
                    ANALYZER.analyze_photo,
                    bytes(image_data),
                    mime_type,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and _is_retryable(exc):
                    await progress.edit_text("Сервис перегружен, пробую ещё раз...")
                    await asyncio.sleep(1.5)
                    continue
                raise

        if estimate is None:
            raise RuntimeError(f"Analysis failed after retry: {last_exc}")

        feedback = _meal_feedback(estimate)
        if user is not None and chat is not None:
            await asyncio.to_thread(append_meal, user.id, chat.id, estimate, _normalize_meal_type(selected_meal_type))
            main_count = await asyncio.to_thread(main_meals_count_for_day, user.id, _today_local())
            target_today = _target_meals_for_day(user.id, _today_local())
            meal_kind = _meal_type_label(selected_meal_type)
            result = (
                f"{_format_estimate(estimate)}\n\n"
                f"Сохранено как: {meal_kind}\n"
                f"{_format_today_progress(main_count, target_today)}\n\n"
                f"{feedback}"
            )
        else:
            result = f"{_format_estimate(estimate)}\n\n{feedback}"

        await progress.edit_text(result)
    except Exception as exc:
        logger.exception("Failed to analyze food photo: %s", exc)
        await progress.edit_text(f"Не получилось обработать фото.\nПричина: {user_friendly_error(exc)}")


async def _analyze_food_text(
    *,
    message,
    user_id: int | None,
    chat_id: int | None,
    description: str,
    meal_type: str = MEAL_TYPE_MEAL,
) -> None:
    payload = (description or "").strip()
    if len(payload) < 3:
        await message.reply_text("Описание слишком короткое. Пример: гречка 200г, курица 150г, салат.")
        return

    progress = await message.reply_text("Оцениваю описание блюда...")
    try:
        estimate: MacroEstimate | None = None
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                estimate = await asyncio.to_thread(ANALYZER.analyze_description, payload)
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and _is_retryable(exc):
                    await progress.edit_text("Сервис перегружен, пробую ещё раз...")
                    await asyncio.sleep(1.5)
                    continue
                raise

        if estimate is None:
            raise RuntimeError(f"Text analysis failed after retry: {last_exc}")

        feedback = _meal_feedback(estimate)
        if user_id is not None and chat_id is not None:
            normalized_meal_type = _normalize_meal_type(meal_type)
            await asyncio.to_thread(append_meal, user_id, chat_id, estimate, normalized_meal_type)
            daily = await asyncio.to_thread(main_meals_count_for_day, user_id, _today_local())
            target_today = _target_meals_for_day(user_id, _today_local())
            meal_kind = _meal_type_label(normalized_meal_type)
            result = (
                f"{_format_estimate(estimate)}\n\n"
                f"Сохранено как: {meal_kind}\n"
                f"{_format_today_progress(daily, target_today)}\n\n"
                f"{feedback}"
            )
        else:
            result = f"{_format_estimate(estimate)}\n\n{feedback}"
        await progress.edit_text(result)
    except Exception as exc:
        logger.exception("Failed to analyze meal description: %s", exc)
        await progress.edit_text(
            "Не получилось разобрать описание блюда.\n"
            f"Причина: {user_friendly_error(exc)}\n"
            "Попробуй написать подробнее: состав и примерный вес/объём."
        )


async def meal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if message is None:
        return
    _set_next_meal_type(context, None)

    if not context.args:
        await message.reply_text(
            "Используй: /meal описание\n"
            "Пример: /meal овсянка на молоке 300г, банан 1 шт, орехи 15г"
        )
        return

    await _analyze_food_text(
        message=message,
        user_id=user.id if user is not None else None,
        chat_id=chat.id if chat is not None else None,
        description=" ".join(context.args).strip(),
        meal_type=MEAL_TYPE_MEAL,
    )


async def snack_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if message is None:
        return

    if not context.args:
        _set_next_meal_type(context, MEAL_TYPE_SNACK)
        await message.reply_text(
            "Режим перекуса включен. Пришли фото или текст.\n"
            "Пример: /snack йогурт 150г и яблоко."
        )
        return

    _set_next_meal_type(context, None)
    await _analyze_food_text(
        message=message,
        user_id=user.id if user is not None else None,
        chat_id=chat.id if chat is not None else None,
        description=" ".join(context.args).strip(),
        meal_type=MEAL_TYPE_SNACK,
    )


async def _handle_pending_input(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    pending = str(context.user_data.get(PENDING_INPUT_KEY) or "").strip()
    if not pending:
        return False

    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return False

    handled = True
    if pending == PENDING_SET_WEIGHT:
        ok = await _apply_weight_raw(message, user.id, text)
    elif pending == PENDING_SET_GOAL:
        ok = await _apply_goal_raw(message, user.id, text)
    elif pending == PENDING_SET_TARGET_DATE:
        ok = await _apply_target_date_raw(message, user.id, text)
    elif pending == PENDING_SET_ACTIVITY_KCAL:
        ok = await _apply_activity_kcal_raw(message, user.id, text)
    elif pending == PENDING_SET_BIRTHDATE:
        ok = await _apply_birthdate_raw(message, user.id, text)
    elif pending == PENDING_DELETE_MEAL_TODAY:
        ok = await _apply_delete_meal_today_raw(message, user.id, text)
    elif pending == PENDING_DELETE_WEIGHT_TODAY:
        ok = await _apply_delete_weight_today_raw(message, user.id, text)
    elif pending == PENDING_SKIP_MEAL_TODAY:
        ok = await _apply_skip_meal_today_raw(message, user.id, text)
    else:
        handled = False
        ok = False

    if handled and ok:
        _set_pending_input(context, None)
    elif not handled:
        _set_pending_input(context, None)
    return handled


async def _handle_menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    message = update.effective_message
    user = update.effective_user
    if message is None:
        return False
    legacy_menu_aliases = {
        "История 14": MENU_BTN_HISTORY,
        "Напоминания ВКЛ": MENU_BTN_REMINDERS_ON,
        "Напоминания ВЫКЛ": MENU_BTN_REMINDERS_OFF,
    }
    text = legacy_menu_aliases.get(text, text)

    if text == MENU_BTN_MEAL:
        _set_pending_input(context, None)
        _set_next_meal_type(context, None)
        await message.reply_text(
            "Отправь фото еды или текст через /meal.\n"
            "Пример: /meal рис 180г, лосось 120г, салат.",
            reply_markup=_main_menu_markup(),
        )
        return True
    if text == MENU_BTN_SNACK:
        _set_pending_input(context, None)
        _set_next_meal_type(context, MEAL_TYPE_SNACK)
        await message.reply_text(
            "Режим перекуса включен. Пришли фото или текст описания перекуса.\n"
            "Пример: /snack йогурт 150г и яблоко.",
            reply_markup=_main_menu_markup(),
        )
        return True
    if text == MENU_BTN_WEIGHT:
        _set_pending_input(context, PENDING_SET_WEIGHT)
        await message.reply_text("Пришли вес на сегодня в кг. Пример: 82.4", reply_markup=_main_menu_markup())
        return True
    if text == MENU_BTN_GOAL:
        _set_pending_input(context, PENDING_SET_GOAL)
        await message.reply_text("Сколько кг хочешь сбросить? Пример: 8.5", reply_markup=_main_menu_markup())
        return True
    if text == MENU_BTN_TARGET_DATE:
        _set_pending_input(context, PENDING_SET_TARGET_DATE)
        await message.reply_text(
            "Пришли целевую дату в формате YYYY-MM-DD. Пример: 2026-09-01",
            reply_markup=_main_menu_markup(),
        )
        return True
    if text == MENU_BTN_ACTIVITY_KCAL:
        _set_pending_input(context, PENDING_SET_ACTIVITY_KCAL)
        await message.reply_text("Сколько ккал сжег за активность? Пример: 420", reply_markup=_main_menu_markup())
        return True
    if text == MENU_BTN_DELETE_MEAL:
        _set_pending_input(context, PENDING_DELETE_MEAL_TODAY)
        if user is None:
            return True
        await message.reply_text(await _format_meals_today_for_delete(user.id), reply_markup=_main_menu_markup())
        return True
    if text == MENU_BTN_DELETE_WEIGHT:
        _set_pending_input(context, PENDING_DELETE_WEIGHT_TODAY)
        if user is None:
            return True
        await message.reply_text(await _format_weights_today_for_delete(user.id), reply_markup=_main_menu_markup())
        return True
    if text == MENU_BTN_SKIP_MEAL:
        _set_pending_input(context, PENDING_SKIP_MEAL_TODAY)
        await message.reply_text(
            "Какой прием пропускаешь сегодня? Напиши: завтрак, обед или ужин (или 1/2/3).",
            reply_markup=_main_menu_markup(),
        )
        return True
    if text == MENU_BTN_ACTIVITY_PHOTO:
        _set_pending_input(context, None)
        await activity_photo_command(update, context)
        return True
    if text == MENU_BTN_TODAY:
        _set_pending_input(context, None)
        await today_command(update, context)
        return True
    if text == MENU_BTN_ADVICE:
        _set_pending_input(context, None)
        await advice_command(update, context)
        return True
    if text == MENU_BTN_PROFILE:
        _set_pending_input(context, None)
        await profile_command(update, context)
        return True
    if text == MENU_BTN_HISTORY:
        _set_pending_input(context, None)
        if user is None:
            return True
        await message.reply_text(_format_history_summary(user.id, 14), reply_markup=_main_menu_markup())
        return True
    if text == MENU_BTN_STATS:
        _set_pending_input(context, None)
        await stats_command(update, context)
        return True
    if text == MENU_BTN_REMINDERS:
        _set_pending_input(context, None)
        await reminders_status_command(update, context)
        return True
    if text == MENU_BTN_REMINDERS_ON:
        _set_pending_input(context, None)
        await reminders_on_command(update, context)
        return True
    if text == MENU_BTN_REMINDERS_OFF:
        _set_pending_input(context, None)
        await reminders_off_command(update, context)
        return True
    if text == MENU_BTN_HELP:
        _set_pending_input(context, None)
        await help_command(update, context)
        return True
    return False


async def text_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    text = str(message.text or "").strip()

    if await _handle_menu_button(update, context, text):
        return

    if await _handle_pending_input(update, context, text):
        return

    queued_meal_type = _peek_next_meal_type(context) or MEAL_TYPE_MEAL
    if _is_trivial_text_message(text):
        if queued_meal_type == MEAL_TYPE_SNACK:
            await message.reply_text(
                "Жду данные по перекусу: пришли фото или текст описания.",
                reply_markup=_main_menu_markup(),
            )
            return
        await message.reply_text(
            "Можешь прислать фото еды или текстовое описание.\n"
            "Пример: /meal рис 180г, лосось 120г, салат.",
            reply_markup=_main_menu_markup(),
        )
        return

    user = update.effective_user
    chat = update.effective_chat
    meal_type = _consume_next_meal_type(context) or MEAL_TYPE_MEAL
    await _analyze_food_text(
        message=message,
        user_id=user.id if user is not None else None,
        chat_id=chat.id if chat is not None else None,
        description=text,
        meal_type=meal_type,
    )


def validate_env() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
    if not ANALYZER.api_keys:
        raise RuntimeError("Missing GEMINI_API_KEYS in .env")


def ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def main() -> None:
    validate_env()
    ensure_event_loop()
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("profile", profile_command))
    app.add_handler(CommandHandler("set_birthdate", set_birthdate_command))
    app.add_handler(CommandHandler("set_goal", set_goal_command))
    app.add_handler(CommandHandler("set_target_date", set_target_date_command))
    app.add_handler(CommandHandler("set_weight", set_weight_command))
    app.add_handler(CommandHandler("advice", advice_command))
    app.add_handler(CommandHandler("meal", meal_command))
    app.add_handler(CommandHandler("snack", snack_command))
    app.add_handler(CommandHandler("meals_today", meals_today_command))
    app.add_handler(CommandHandler("delete_meal_today", delete_meal_today_command))
    app.add_handler(CommandHandler("weights_today", weights_today_command))
    app.add_handler(CommandHandler("delete_weight_today", delete_weight_today_command))
    app.add_handler(CommandHandler("skip_meal", skip_meal_command))
    app.add_handler(CommandHandler("unskip_meal", unskip_meal_command))
    app.add_handler(CommandHandler("skips_today", skips_today_command))
    app.add_handler(CommandHandler("activity_photo", activity_photo_command))
    app.add_handler(CommandHandler("set_activity_kcal", set_activity_kcal_command))
    app.add_handler(CommandHandler("clear_goal", clear_goal_command))
    app.add_handler(CommandHandler("reminders", reminders_status_command))
    app.add_handler(CommandHandler("reminders_on", reminders_on_command))
    app.add_handler(CommandHandler("reminders_off", reminders_off_command))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, analyze_food_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_fallback))
    schedule_reminders(app)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
