# Goal Habits Bot (MVP)

Telegram bot MVP for your personal goals project.

Current POC feature:
- send a food photo
- bot estimates calories and macros (protein/fat/carbs) with Gemini
- bot shows an uncertainty range (error margin)
- bot tracks meal count and totals for today (`/today`)
- bot keeps long-term history and aggregate stats (`/history`, `/stats`)
- user profile settings saved per user (`/profile`, `/set_birthdate`, `/set_goal`)
- personalized daily advice (`/advice`) based on age, morning weight, goal, and target date
- optional scheduled reminders (3 times/day by default)
- evening activity intake via screenshot/manual value, included in advice
- chat menu with buttons for core actions (`/menu`)
- skip meal slots for today (`/skip_meal`, `/skips_today`)
- delete today's meal/weight entries directly in Telegram

## Setup

```powershell
cd C:\Users\Gena\Documents\Playground\goal_habits_bot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure

```powershell
Copy-Item .env.example .env
```

`.env` fields:
- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEYS` (one or multiple keys separated by `,` or `;`)
- `GEMINI_MODEL` (default: `gemini-3.1-flash-lite-preview`)
- `GEMINI_TIMEOUT_SEC`
- `TARGET_MEALS_PER_DAY` (default: `3`)
- `MEAL_LOG_PATH` (default: `data/meal_log.jsonl`, persistent between restarts)
- `MEAL_SKIP_PATH` (default: `data/meal_skip.json`, per-user skipped meal slots)
- `USER_SETTINGS_PATH` (default: `data/user_settings.json`, persistent per-user settings)
- `WEIGHT_LOG_PATH` (default: `data/weight_log.jsonl`, per-user morning weights)
- `ACTIVITY_LOG_PATH` (default: `data/activity_log.jsonl`, per-user activity calories)
- `BOT_TIMEZONE` (default: `Europe/Belgrade`)
- `REMINDER_TIMES` (default: `09:00,14:00,20:00`)
- `REMINDER_TEXT` (text for scheduled reminder messages)
- `ACTIVITY_REMINDER_TIME` (default: `21:30`)
- `ACTIVITY_REMINDER_TEXT` (text for evening activity reminder)

## Run

```powershell
python bot.py
```

## Commands

- `/start`
- `/help`
- `/model`
- `/menu`
- `/today`
- `/history`
- `/stats`
- `/profile`
- `/set_birthdate`
- `/set_goal`
- `/set_target_date`
- `/set_weight`
- `/advice`
- `/meal`
- `/skip_meal`
- `/unskip_meal`
- `/skips_today`
- `/meals_today`
- `/delete_meal_today`
- `/weights_today`
- `/delete_weight_today`
- `/activity_photo`
- `/set_activity_kcal`
- `/clear_goal`
- `/reminders`
- `/reminders_on`
- `/reminders_off`

## Notes

- This is a visual estimate, not a medical-grade nutrition calculation.
- For better quality: send a clear photo, one meal per image, visible portion size.
