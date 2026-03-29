from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from meal_log import DailySummary


@dataclass
class AdviceResult:
    daily_calorie_target: float
    daily_protein_target: float
    daily_fat_target: float
    daily_carbs_target: float
    remaining_calories: float
    burned_activity_kcal_today: float
    net_consumed_kcal_today: float
    remaining_protein: float
    remaining_fat: float
    remaining_carbs: float
    maintenance_estimate: float
    deficit_required_per_day: float
    deficit_used_per_day: float
    days_left: int
    feasibility_warning: str | None
    meal_suggestion: str


def _maintenance_factor_by_age(age: int) -> float:
    # Conservative sedentary baseline (without logged activity).
    # We add explicit activity calories separately from activity logs/screens.
    if age < 30:
        return 24.0
    if age < 45:
        return 22.0
    if age < 60:
        return 21.0
    return 20.0


def _build_meal_suggestion(rem_p: float, rem_f: float, rem_c: float, rem_kcal: float) -> str:
    if rem_kcal <= 120:
        return "Лёгкий вариант: овощной салат и вода/чай без сахара."

    chicken_g = max(0.0, rem_p / 0.31)  # ~31g protein / 100g
    rice_g = max(0.0, rem_c / 0.28)  # ~28g carbs / 100g cooked
    oil_g = max(0.0, rem_f)  # ~1g fat / 1g oil

    chicken_g = min(chicken_g, 260.0)
    rice_g = min(rice_g, 280.0)
    oil_g = min(oil_g, 15.0)

    if rem_p < 15 and rem_c < 20:
        return "Небольшой перекус: греческий йогурт 150-200 г и ягоды."

    return (
        "Пример приёма пищи: "
        f"куриная грудка ~{chicken_g:.0f} г, "
        f"рис/гречка в готовом виде ~{rice_g:.0f} г, "
        f"овощи 200-300 г, "
        f"оливковое масло ~{oil_g:.0f} г."
    )


def build_advice(
    *,
    age: int,
    weight_kg: float,
    goal_loss_kg: float,
    target_date: date,
    today: date,
    consumed_today: DailySummary,
    burned_activity_kcal_today: float = 0.0,
) -> AdviceResult:
    if age <= 0:
        raise ValueError("Age must be positive")
    if weight_kg <= 0:
        raise ValueError("Weight must be positive")
    if goal_loss_kg <= 0:
        raise ValueError("Goal loss must be positive")

    days_left = (target_date - today).days
    if days_left <= 0:
        raise ValueError("Target date must be in the future")

    maintenance = weight_kg * _maintenance_factor_by_age(age)
    required_deficit = (goal_loss_kg * 7700.0) / float(days_left)

    max_safe_deficit = min(1000.0, max(350.0, weight_kg * 11.0))
    used_deficit = min(required_deficit, max_safe_deficit)
    feasibility_warning: str | None = None
    if required_deficit > max_safe_deficit:
        feasibility_warning = (
            "Текущая цель слишком агрессивна для безопасного темпа. "
            "Расчёт выполнен с ограничением дефицита."
        )

    min_calories = max(1200.0, weight_kg * 18.0)
    calories_target = maintenance - used_deficit
    if calories_target < min_calories:
        calories_target = min_calories
        feasibility_warning = (
            feasibility_warning or "Для безопасности дневная цель калорий повышена до минимально разумной."
        )

    protein_target = max(70.0, weight_kg * 1.6)
    fat_target = max(40.0, weight_kg * 0.8)
    carbs_target = max(50.0, (calories_target - protein_target * 4.0 - fat_target * 9.0) / 4.0)

    burned_kcal = max(0.0, float(burned_activity_kcal_today or 0.0))
    net_consumed_kcal = consumed_today.total_kcal - burned_kcal
    remaining_calories = calories_target - net_consumed_kcal
    remaining_protein = protein_target - consumed_today.total_protein_g
    remaining_fat = fat_target - consumed_today.total_fat_g
    remaining_carbs = carbs_target - consumed_today.total_carbs_g

    meal_hint = _build_meal_suggestion(
        rem_p=max(0.0, remaining_protein),
        rem_f=max(0.0, remaining_fat),
        rem_c=max(0.0, remaining_carbs),
        rem_kcal=max(0.0, remaining_calories),
    )

    return AdviceResult(
        daily_calorie_target=calories_target,
        daily_protein_target=protein_target,
        daily_fat_target=fat_target,
        daily_carbs_target=carbs_target,
        remaining_calories=remaining_calories,
        burned_activity_kcal_today=burned_kcal,
        net_consumed_kcal_today=net_consumed_kcal,
        remaining_protein=remaining_protein,
        remaining_fat=remaining_fat,
        remaining_carbs=remaining_carbs,
        maintenance_estimate=maintenance,
        deficit_required_per_day=required_deficit,
        deficit_used_per_day=used_deficit,
        days_left=days_left,
        feasibility_warning=feasibility_warning,
        meal_suggestion=meal_hint,
    )
