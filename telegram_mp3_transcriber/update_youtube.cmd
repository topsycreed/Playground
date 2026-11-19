@echo off
setlocal
set "BOT_PYTHON=%~dp0.venv312\Scripts\python.exe"
if not exist "%BOT_PYTHON%" (
    echo Bot Python was not found at "%BOT_PYTHON%".
    exit /b 1
)
"%BOT_PYTHON%" -m pip install --upgrade --pre "yt-dlp[default]"
if errorlevel 1 exit /b 1
"%BOT_PYTHON%" -m yt_dlp --version
echo Restart python bot.py to use the updated YouTube downloader.
