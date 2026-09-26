"""
signal_checker.py — Проверка сигналов и отправка уведомлений в Telegram.
Запускается по расписанию (например, через GitHub Actions).
"""

import os
import glob
import pandas as pd
import joblib
from telegram import Bot
from dotenv import load_dotenv
import config

load_dotenv()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def load_models():
    """Загружает обученные модели из папки models/."""
    models = {}
    for regime in ["BULL", "BEAR", "NEUTRAL"]:
        fname = os.path.join(config.MODELS_DIR, f"model_{regime}.joblib")
        if os.path.exists(fname):
            models[regime] = joblib.load(fname)
    return models if models else None


def get_latest_signals():
    """Возвращает список текущих сигналов на покупку."""
    parquet_files = sorted(
        glob.glob(os.path.join(config.DATA_PROCESSED, "market_ml_ready_*.parquet")),
        reverse=True
    )
    if not parquet_files:
        return [], "Нет подготовленных данных."

    df = pd.read_parquet(parquet_files[0])
    last_date = df["Date"].max()
    df_last = df[df["Date"] == last_date].copy()
    if df_last.empty:
        return [], "Нет данных на последнюю дату."

    models = load_models()
    if not models:
        return [], "Модели не найдены."

    signals = []
    for _, row in df_last.iterrows():
        if row["bounce_long_signal"] != 1:
            continue
        if row["regime"] == "BEAR":
            continue
        if row["vol_regime"] != "calm":
            continue

        regime = row["regime"]
        if regime not in models:
            regime = "NEUTRAL" if "NEUTRAL" in models else list(models.keys())[0]

        X = pd.DataFrame([row[config.FEATURE_COLS].values], columns=config.FEATURE_COLS)
        proba = models[regime].predict_proba(X)[0, 1]

        if proba >= config.PROBA_LONG:
            signals.append({
                "ticker": row["Ticker"],
                "close": row["close"],
                "proba": proba,
                "rsi": row["rsi"],
            })

    return signals, None


def main():
    if not TOKEN or not CHAT_ID:
        print("Ошибка: не заданы TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID.")
        return

    print("Проверка сигналов...")
    signals, error = get_latest_signals()

    if error:
        print(f"Ошибка: {error}")
        return

    if not signals:
        print("Сигналов нет.")
        return

    bot = Bot(token=TOKEN)
    lines = ["📈 Сигналы на покупку:\n"]
    for s in signals:
        lines.append(
            f"• {s['ticker']} — цена: {s['close']:.2f}, "
            f"вероятность: {s['proba']:.1%}, RSI: {s['rsi']:.1f}"
        )
    message = "\n".join(lines)

    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
        print("Сигналы отправлены.")
    except Exception as e:
        print(f"Ошибка при отправке: {e}")


if __name__ == "__main__":
    main()
