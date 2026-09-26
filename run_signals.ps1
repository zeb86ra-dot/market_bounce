# run_signals.ps1 — автоматический запуск live_signals.py с Telegram-уведомлениями

# === Пути ===
$projectRoot = "C:\Users\User\market_bounce"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $projectRoot "live_signals.py"

# === Telegram-настройки ===
# Замените на свои значения!
$env:TG_TOKEN = "8844038055:AAFE1tni5O2PDspSaHhboI5xW4NKynXHk84"
$env:TG_CHAT_ID = "6926174034"

# === Запуск ===
& $python $script
