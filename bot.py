"""
bot.py — Telegram-бот для управления bounce-стратегией
"""

import os
import subprocess
import sys
import glob
from datetime import datetime
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

import config

# Загружаем переменные из .env файла
load_dotenv()

# Получаем токен из окружения или из .env, убираем возможные пробелы/переносы
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# Если токен пустой – выводим сообщение и завершаем работу
if not TOKEN:
    print("Ошибка: укажите токен бота в файле .env (TELEGRAM_BOT_TOKEN=...) или в переменной окружения.")
    sys.exit(1)

LOG_DIR = getattr(config, "LOG_DIR", "logs")
DATA_RESULTS = config.DATA_RESULTS
GRAPHS_DIR = config.GRAPHS_DIR

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_RESULTS, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


def run_script(script_name):
    """Запускает python-скрипт и возвращает (returncode, log_file)"""
    log_file = os.path.join(LOG_DIR, f"{script_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, script_name],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
    return result.returncode, log_file


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для управления bounce-стратегией.\n\n"
        "Доступные команды:\n"
        "/fetch - загрузить данные с MOEX\n"
        "/prepare - подготовить признаки\n"
        "/strategy - запустить бэктест\n"
        "/run - выполнить все этапы\n"
        "/results - показать последние результаты\n"
        "/help - помощь"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n"
        "/fetch - загрузить свежие данные\n"
        "/prepare - пересчитать признаки\n"
        "/strategy - запустить стратегию\n"
        "/run - всё сразу (fetch -> prepare -> strategy)\n"
        "/results - метрики и последние сделки"
    )


async def fetch_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Загружаю данные... Это может занять несколько минут.")
    code, log = run_script("fetch_data.py")
    if code == 0:
        await update.message.reply_text("Данные загружены успешно.")
    else:
        await update.message.reply_text(f"Ошибка загрузки. Лог: {log}")


async def prepare_features(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Готовлю признаки...")
    code, log = run_script("prepare_features.py")
    if code == 0:
        await update.message.reply_text("Признаки готовы.")
    else:
        await update.message.reply_text(f"Ошибка подготовки. Лог: {log}")


async def run_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Запускаю стратегию...")
    code, log = run_script("strategy_bounce_v2.py")
    if code == 0:
        await update.message.reply_text("Стратегия выполнена. Результаты скоро пришлю.")
        await send_latest_results(update)
    else:
        await update.message.reply_text(f"Ошибка стратегии. Лог: {log}")


async def run_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Запускаю полный цикл...")
    steps = ["fetch_data.py", "prepare_features.py", "strategy_bounce_v2.py"]
    for step in steps:
        await update.message.reply_text(f"Выполняю {step}...")
        code, log = run_script(step)
        if code != 0:
            await update.message.reply_text(f"Ошибка на шаге {step}. Лог: {log}")
            return
    await update.message.reply_text("Все этапы завершены успешно.")
    await send_latest_results(update)


async def send_latest_results(update: Update):
    """Отправляет последние метрики и, при наличии, график"""
    equity_files = sorted(glob.glob(os.path.join(DATA_RESULTS, "equity_*.csv")), reverse=True)
    if not equity_files:
        await update.message.reply_text("Нет файлов с результатами.")
        return

    eq = pd.read_csv(equity_files[0])
    daily_rets = eq["daily_return"]
    total_return = (1 + daily_rets).prod() - 1
    sharpe = daily_rets.mean() / daily_rets.std() * (252**0.5) if daily_rets.std() > 0 else 0

    trades_file = equity_files[0].replace("equity", "trades")
    if os.path.exists(trades_file):
        trades_df = pd.read_csv(trades_file)
        num_trades = len(trades_df)
    else:
        num_trades = "?"

    text = (
        f"Последние результаты:\n"
        f"Общая доходность: {total_return:.2%}\n"
        f"Sharpe: {sharpe:.2f}\n"
        f"Сделок: {num_trades}\n"
    )
    await update.message.reply_text(text)

    graphs = sorted(glob.glob(os.path.join(GRAPHS_DIR, "equity_curve_*.png")), reverse=True)
    if graphs:
        with open(graphs[0], "rb") as photo:
            await update.message.reply_photo(photo, caption="Кривая капитала")


async def show_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_latest_results(update)


def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("fetch", fetch_data))
    application.add_handler(CommandHandler("prepare", prepare_features))
    application.add_handler(CommandHandler("strategy", run_strategy))
    application.add_handler(CommandHandler("run", run_all))
    application.add_handler(CommandHandler("results", show_results))

    print("Бот запущен...")
    application.run_polling()


if __name__ == "__main__":
    main()