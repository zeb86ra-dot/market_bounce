"""
live_signals.py — генератор реальных торговых сигналов v8
+ Логирование в файл
+ Проверка свежести данных
+ Дедупликация сигналов (хеш последних сигналов)
+ Уведомления в Telegram при ошибках и новых сигналах
+ Обработка ошибок на каждом этапе

Использование:
    python live_signals.py              — обычный запуск, вывод в консоль
    python live_signals.py --days 10    — сигналы за последние 10 дней
    python live_signals.py --all        — только активные позиции
    python live_signals.py --no-notify  — не отправлять Telegram-уведомления
"""

import os
import sys
import glob
import json
import hashlib
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from config import *

# === Фильтры (идентичны strategy_bounce_v2.py v7) ===
BAD_TICKERS = {"GMKN", "NVTK", "SBER", "CHMF", "PLZL", "TATN", "YDEX", "VTBR"}
COOLDOWN_DAYS = 3
MIN_SCORE = 0.0
DIRECTIONS_ACTIVE = ["short"]

# === Telegram ===
TELEGRAM_TOKEN = os.environ.get("TG_TOKEN", "")        # токен бота
TELEGRAM_CHAT_ID = os.environ.get("TG_CHAT_ID", "")     # chat_id
DATA_STALE_DAYS = 5                                      # порог устаревания

# === Пути ===
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LAST_SIGNALS_FILE = os.path.join(PROJECT_ROOT, "logs", "last_signals_hash.txt")
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
class Logger:
    """Двойной лог: в файл и в консоль."""

    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_DIR, f"signals_{ts}.log")
        self._buffer = []

    def log(self, msg=""):
        line = str(msg)
        self._buffer.append(line)
        print(line)

    def flush(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self._buffer))
        return self.log_file


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(text):
    """Отправка сообщения в Telegram. Молча игнорирует, если не настроен."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠️  Telegram error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"  ⚠️  Telegram не отправлен: {e}")


# ---------------------------------------------------------------------------
# Дедупликация
# ---------------------------------------------------------------------------
def compute_signal_hash(signals):
    """Считает хеш от списка новых сигналов."""
    if not signals:
        return ""
    lines = []
    for s in signals:
        lines.append(f"{s['ticker']}|{s['entry_price']:.2f}|{s['date'].strftime('%Y-%m-%d')}")
    raw = "|".join(sorted(lines))
    return hashlib.md5(raw.encode()).hexdigest()


def is_duplicate(new_hash):
    """Проверяет, отправлялся ли уже такой набор сигналов."""
    if not new_hash:
        return True
    try:
        if os.path.exists(LAST_SIGNALS_FILE):
            with open(LAST_SIGNALS_FILE, "r") as f:
                old_hash = f.read().strip()
            return old_hash == new_hash
    except Exception:
        pass
    return False


def save_hash(new_hash):
    """Сохраняет хеш текущих сигналов."""
    try:
        with open(LAST_SIGNALS_FILE, "w") as f:
            f.write(new_hash)
    except Exception as e:
        print(f"  ⚠️  Не удалось сохранить хеш: {e}")


# ---------------------------------------------------------------------------
# Загрузка и проверка данных
# ---------------------------------------------------------------------------
def load_data(logger):
    """Загружает enriched-датасет. Бросает исключение при ошибке."""
    files = sorted(glob.glob(
        os.path.join(DATA_PROCESSED, "bounce_enriched_*.parquet")
    ))
    files = [f for f in files if "with_target" not in f]
    if not files:
        raise FileNotFoundError(
            "Нет enriched данных в data/processed/. Запустите build_features.py"
        )
    logger.log(f"  Датасет: {os.path.basename(files[-1])}")
    df = pd.read_parquet(files[-1])
    df.columns = [str(c).lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("В датасете нет колонки 'date'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def check_data_freshness(df, logger):
    """Проверяет, насколько свежие данные. Возвращает (is_stale, days_old)."""
    last_date = df["date"].max()
    today = datetime.now()
    days_old = (today - pd.Timestamp(last_date)).days
    logger.log(f"\nПоследняя свеча в датасете: {last_date.strftime('%Y-%m-%d')}")
    logger.log(f"Сегодня: {today.strftime('%Y-%m-%d %H:%M')}")

    if days_old > DATA_STALE_DAYS:
        logger.log(f"\n⚠️  Данные устарели на {days_old} дней!")
        logger.log("   Обновите свечи: python fetch_data.py")
        logger.log("   Затем пересчитайте: python build_features.py")
        return True, days_old
    return False, days_old


# ---------------------------------------------------------------------------
# Проверка входа
# ---------------------------------------------------------------------------
def check_entry(grp, i):
    """Проверяет, есть ли сигнал входа на свече i. Возвращает (direction, score) или None."""
    if pd.isna(grp["atr_14"].iloc[i]) or grp["atr_14"].iloc[i] == 0:
        return None

    score_long = grp["bounce_score_long"].iloc[i] if "bounce_score_long" in grp.columns else 0
    score_short = grp["bounce_score_short"].iloc[i] if "bounce_score_short" in grp.columns else 0

    direction = None
    score = 0.0

    if "long" in DIRECTIONS_ACTIVE and score_long > MIN_SCORE and score_long >= score_short:
        direction = "long"
        score = score_long
    elif "short" in DIRECTIONS_ACTIVE and score_short > MIN_SCORE and score_short > score_long:
        direction = "short"
        score = score_short

    if direction is None:
        return None

    # Трендовый фильтр
    if USE_TREND_FILTER and "ma_200" in grp.columns:
        ma200 = grp["ma_200"].iloc[i]
        close_i = grp["close"].iloc[i]
        if not pd.isna(ma200):
            if direction == "long" and close_i < ma200:
                return None
            if direction == "short" and close_i > ma200:
                return None

    return (direction, score)


# ---------------------------------------------------------------------------
# Симуляция сделки
# ---------------------------------------------------------------------------
def simulate_trade_from_entry(grp, entry_idx, direction):
    """Симулирует сделку от entry_idx. Возвращает статус: open / closed + детали."""
    entry_price = grp["close"].iloc[entry_idx]
    atr_value = grp["atr_14"].iloc[entry_idx]
    stop_dist = ATR_MULT_STOP * atr_value
    take_dist = stop_dist * RR_RATIO

    if direction == "long":
        stop_price = entry_price - stop_dist
        take_price = entry_price + take_dist
    else:
        stop_price = entry_price + stop_dist
        take_price = entry_price - take_dist

    current_stop = stop_price
    trail_activated = False
    end_idx = min(entry_idx + MAX_HOLDING_DAYS, len(grp) - 1)

    for i in range(entry_idx + 1, end_idx + 1):
        high = grp["high"].iloc[i]
        low = grp["low"].iloc[i]
        close = grp["close"].iloc[i]
        atr_i = grp["atr_14"].iloc[i] if not pd.isna(grp["atr_14"].iloc[i]) else atr_value

        if direction == "long" and low <= current_stop:
            return "closed", "stop", current_stop, i, i - entry_idx
        if direction == "short" and high >= current_stop:
            return "closed", "stop", current_stop, i, i - entry_idx

        if direction == "long" and high >= take_price:
            return "closed", "take", take_price, i, i - entry_idx
        if direction == "short" and low <= take_price:
            return "closed", "take", take_price, i, i - entry_idx

        if not trail_activated:
            if direction == "long":
                if close - entry_price >= TRAIL_AFTER_RR * stop_dist:
                    trail_activated = True
            else:
                if entry_price - close >= TRAIL_AFTER_RR * stop_dist:
                    trail_activated = True

        if trail_activated:
            if direction == "long":
                new_stop = close - TRAIL_ATR_MULT * atr_i
                current_stop = max(current_stop, new_stop)
            else:
                new_stop = close + TRAIL_ATR_MULT * atr_i
                current_stop = min(current_stop, new_stop)

    last_idx = len(grp) - 1
    if end_idx == last_idx:
        return "open", "active", current_stop, last_idx, last_idx - entry_idx

    last_close = grp["close"].iloc[end_idx]
    return "closed", "timeout", last_close, end_idx, end_idx - entry_idx


# ---------------------------------------------------------------------------
# Расчёт позиции
# ---------------------------------------------------------------------------
def calc_position_size(entry_price, stop_price, risk_pct=1.0):
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share == 0:
        return 0
    capital = 100_000
    risk_amount = capital * risk_pct / 100
    shares = int(risk_amount / risk_per_share)
    return shares


# ---------------------------------------------------------------------------
# Форматирование сигнала
# ---------------------------------------------------------------------------
def format_signal(ticker, direction, entry_date, entry_price, stop_price,
                  take_price, atr_val, score, status, current_stop=None,
                  current_price=None, holding_days=0):
    direction_ru = "ШОРТ" if direction == "short" else "ЛОНГ"
    risk_pct = abs(entry_price - stop_price) / entry_price * 100
    reward_pct = abs(take_price - entry_price) / entry_price * 100
    rr = reward_pct / risk_pct if risk_pct > 0 else 0
    shares = calc_position_size(entry_price, stop_price)

    lines = []
    lines.append(f"  {'─' * 54}")
    lines.append(f"  📌 {ticker} — {direction_ru}")
    lines.append(f"     Дата входа:     {entry_date.strftime('%Y-%m-%d')}")
    lines.append(f"     Цена входа:     {entry_price:.2f} ₽")
    lines.append(f"     Стоп-лосс:      {stop_price:.2f} ₽  ({risk_pct:.1f}%)")
    lines.append(f"     Тейк-профит:    {take_price:.2f} ₽  (+{reward_pct:.1f}%)")
    lines.append(f"     Risk/Reward:    1:{rr:.1f}")
    lines.append(f"     ATR:            {atr_val:.2f}")
    lines.append(f"     Bounce score:   {score:.4f}")
    lines.append(f"     Рекомендация:   {shares} акций (риск 1% от 100к)")

    if status == "open" and current_stop is not None and current_price is not None:
        lines.append(f"     ⏳ АКТИВНА — {holding_days} дн. в позиции")
        lines.append(f"     Текущая цена:   {current_price:.2f} ₽")
        lines.append(f"     Текущий стоп:   {current_stop:.2f} ₽")
        unrealized = ((current_price - entry_price) / entry_price) if direction == "long" \
                     else ((entry_price - current_price) / entry_price)
        lines.append(f"     PnL:            {unrealized:+.1%}")
    elif status == "closed":
        lines.append(f"     ✅ Закрыта ({holding_days} дн.)")

    return "\n".join(lines)


def format_signal_telegram(s):
    """Краткий формат для Telegram — без лишних строк."""
    direction_ru = "ШОРТ" if s["direction"] == "short" else "ЛОНГ"
    risk_pct = abs(s["entry_price"] - s["stop_price"]) / s["entry_price"] * 100
    reward_pct = abs(s["take_price"] - s["entry_price"]) / s["entry_price"] * 100
    shares = calc_position_size(s["entry_price"], s["stop_price"])
    return (
        f"📌 <b>{s['ticker']} — {direction_ru}</b>\n"
        f"Вход: {s['entry_price']:.2f} ₽\n"
        f"Стоп: {s['stop_price']:.2f} ₽ ({risk_pct:.1f}%)\n"
        f"Тейк: {s['take_price']:.2f} ₽ (+{reward_pct:.1f}%)\n"
        f"Score: {s['score']:.3f} | {shares} акций"
    )


def format_active_telegram(s):
    """Краткий формат активной позиции для Telegram."""
    direction_ru = "ШОРТ" if s["direction"] == "short" else "ЛОНГ"
    pnl = ((s["entry_price"] - s["current_price"]) / s["entry_price"]) \
          if s["direction"] == "short" \
          else ((s["current_price"] - s["entry_price"]) / s["entry_price"])
    return (
        f"⏳ <b>{s['ticker']} — {direction_ru}</b> "
        f"({s['holding_days']} дн.) PnL: {pnl:+.1%}\n"
        f"Вход: {s['entry_price']:.2f} | Цена: {s['current_price']:.2f} | "
        f"Стоп: {s['current_stop']:.2f}"
    )


# ---------------------------------------------------------------------------
# Главный пайплайн
# ---------------------------------------------------------------------------
def run(args, logger):
    """Основная логика. Возвращает (new_signals, active_signals, is_stale, days_old)."""

    # --- Загрузка данных ---
    logger.log("=" * 60)
    logger.log("live_signals.py v8 — генератор сигналов")
    logger.log(f"Стратегия: шорт от resistance, стоп {ATR_MULT_STOP}x ATR, тейк {RR_RATIO}R")
    logger.log(f"Тикеры: {sorted(set(TICKERS) - BAD_TICKERS)}")
    logger.log(f"Трендовый фильтр: {USE_TREND_FILTER}")
    logger.log("=" * 60)

    df = load_data(logger)

    # --- Проверка свежести ---
    is_stale, days_old = check_data_freshness(df, logger)

    # --- Сбор сигналов ---
    active_signals = []
    recent_signals = []

    for ticker, grp in df.groupby("ticker"):
        if ticker in BAD_TICKERS:
            continue

        grp = grp.reset_index(drop=True)
        n = len(grp)
        last_entry_idx = -999

        for i in range(n):
            if i - last_entry_idx < COOLDOWN_DAYS:
                continue

            result = check_entry(grp, i)
            if result is None:
                continue

            direction, score = result
            entry_price = grp["close"].iloc[i]
            atr_val = grp["atr_14"].iloc[i]
            stop_dist = ATR_MULT_STOP * atr_val
            take_dist = stop_dist * RR_RATIO

            if direction == "short":
                stop_price = entry_price + stop_dist
                take_price = entry_price - take_dist
            else:
                stop_price = entry_price - stop_dist
                take_price = entry_price + take_dist

            status, reason, exit_price, exit_idx, hold = simulate_trade_from_entry(
                grp, i, direction
            )

            entry_date = grp["date"].iloc[i]

            signal = {
                "ticker": ticker,
                "direction": direction,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "take_price": take_price,
                "atr_val": atr_val,
                "score": score,
                "status": status,
                "exit_reason": reason,
                "holding_days": hold,
                "current_stop": exit_price if status == "open" else None,
                "current_price": grp["close"].iloc[exit_idx] if status == "open" else None,
            }

            if status == "open":
                active_signals.append(signal)
            else:
                days_since = (df["date"].max() - entry_date).days
                if days_since <= args.days:
                    recent_signals.append(signal)

            last_entry_idx = exit_idx if status == "closed" else i + MAX_HOLDING_DAYS

    # --- Вывод активных сигналов ---
    logger.log(f"\n{'=' * 60}")
    logger.log("🔴 АКТИВНЫЕ СИГНАЛЫ (открытые позиции)")
    logger.log(f"{'=' * 60}")

    if active_signals:
        for s in active_signals:
            logger.log(format_signal(
                s["ticker"], s["direction"], s["entry_date"], s["entry_price"],
                s["stop_price"], s["take_price"], s["atr_val"], s["score"],
                s["status"], s["current_stop"], s["current_price"], s["holding_days"]
            ))
    else:
        logger.log("  Нет активных позиций.")

    # --- Вывод недавних сигналов ---
    if not args.all:
        logger.log(f"\n{'=' * 60}")
        logger.log(f"📋 СИГНАЛЫ ЗА ПОСЛЕДНИЕ {args.days} ДНЕЙ")
        logger.log(f"{'=' * 60}")

        if recent_signals:
            for s in sorted(recent_signals, key=lambda x: x["entry_date"], reverse=True):
                logger.log(format_signal(
                    s["ticker"], s["direction"], s["entry_date"], s["entry_price"],
                    s["stop_price"], s["take_price"], s["atr_val"], s["score"],
                    s["status"]
                ))
        else:
            logger.log("  Нет новых сигналов за указанный период.")

    # --- Свежие сигналы (последняя свеча, без дублирования активных) ---
    logger.log(f"\n{'=' * 60}")
    logger.log("🆕 СИГНАЛЫ НА ПОСЛЕДНЕЙ СВЕЧЕ")
    logger.log(f"{'=' * 60}")

    active_tickers = {s["ticker"] for s in active_signals}
    last_candle_signals = []
    last_date = df["date"].max()

    for ticker, grp in df.groupby("ticker"):
        if ticker in BAD_TICKERS or ticker in active_tickers:
            continue
        grp = grp.reset_index(drop=True)
        i = len(grp) - 1

        result = check_entry(grp, i)
        if result is None:
            continue

        direction, score = result
        entry_price = grp["close"].iloc[i]
        atr_val = grp["atr_14"].iloc[i]
        stop_dist = ATR_MULT_STOP * atr_val

        if direction == "short":
            stop_price = entry_price + stop_dist
            take_price = entry_price - stop_dist * RR_RATIO
        else:
            stop_price = entry_price - stop_dist
            take_price = entry_price + stop_dist * RR_RATIO

        last_candle_signals.append({
            "ticker": ticker,
            "direction": direction,
            "date": grp["date"].iloc[i],
            "entry_price": entry_price,
            "stop_price": stop_price,
            "take_price": take_price,
            "atr_val": atr_val,
            "score": score,
        })

    if last_candle_signals:
        for s in last_candle_signals:
            logger.log(format_signal(
                s["ticker"], s["direction"], s["date"], s["entry_price"],
                s["stop_price"], s["take_price"], s["atr_val"], s["score"],
                "new"
            ))
    else:
        logger.log("  Нет новых сигналов на последней свече.")

    # --- Сводка ---
    logger.log(f"\n{'=' * 60}")
    logger.log("📊 СВОДКА")
    logger.log(f"{'=' * 60}")
    logger.log(f"  Активных позиций:    {len(active_signals)}")
    logger.log(f"  Сигналов за {args.days} дн.:    {len(recent_signals)}")
    logger.log(f"  Свежих сигналов:     {len(last_candle_signals)}")
    logger.log(f"  Тикеров в работе:    {len(set(TICKERS) - BAD_TICKERS)}")
    logger.log(f"  Последняя свеча:     {last_date.strftime('%Y-%m-%d')}")

    if active_signals:
        total_unrealized = 0
        for s in active_signals:
            if s["current_price"] and s["entry_price"]:
                pnl = ((s["current_price"] - s["entry_price"]) / s["entry_price"]) \
                      if s["direction"] == "long" \
                      else ((s["entry_price"] - s["current_price"]) / s["entry_price"])
                total_unrealized += pnl
        logger.log(f"  Нереализованный PnL: {total_unrealized:+.1%}")

    logger.log(f"\n⚠️  Напоминание:")
    logger.log(f"  — Сигналы основаны на дневных свечах, проверяйте после закрытия рынка")
    logger.log(f"  — Стоп и тейк — рыночные ордера на следующий день")
    logger.log(f"  — Риск на сделку: 1% от капитала")
    logger.log(f"  — Стратегия только шорт, только в нисходящем тренде (close < MA200)")

    return last_candle_signals, active_signals, is_stale, days_old


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Генератор торговых сигналов v8")
    parser.add_argument("--days", type=int, default=5,
                        help="Сколько последних торговых дней проверять (default: 5)")
    parser.add_argument("--all", action="store_true",
                        help="Показать только активные (незакрытые) сигналы")
    parser.add_argument("--no-notify", action="store_true",
                        help="Не отправлять Telegram-уведомления")
    args = parser.parse_args()

    logger = Logger()
    send_notify = not args.no_notify

    try:
        # --- Основной пайплайн ---
        new_signals, active_signals, is_stale, days_old = run(args, logger)

        # --- Дедупликация ---
        new_hash = compute_signal_hash(new_signals)
        is_dup = is_duplicate(new_hash)

        # --- Telegram-уведомления ---
        if send_notify:

            # 1. Если данные устарели — предупреждение
            if is_stale:
                send_telegram(
                    f"⚠️ <b>Данные устарели на {days_old} дней!</b>\n"
                    f"Обновите: <code>python fetch_data.py</code> → "
                    f"<code>python build_features.py</code>"
                )

            # 2. Новые сигналы (если не дубликат)
            if new_signals and not is_dup:
                lines = ["🆕 <b>Новые сигналы:</b>\n"]
                for s in new_signals:
                    lines.append(format_signal_telegram(s))
                    lines.append("")
                send_telegram("\n".join(lines))
                save_hash(new_hash)
            elif new_signals and is_dup:
                logger.log("\n  ℹ️  Сигналы не отправлены — дубликат последней отправки.")

            # 3. Активные позиции (краткий статус)
            if active_signals:
                lines = ["⏳ <b>Активные позиции:</b>\n"]
                for s in active_signals:
                    lines.append(format_active_telegram(s))
                send_telegram("\n".join(lines))

            # 4. Если ничего нет
            if not new_signals and not active_signals and not is_stale:
                send_telegram("📭 Новых сигналов нет. Активных позиций нет.")

    except FileNotFoundError as e:
        logger.log(f"\n❌ ОШИБКА: {e}")
        if send_notify:
            send_telegram(f"❌ <b>Ошибка:</b> {e}")
    except Exception as e:
        logger.log(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА:")
        logger.log(traceback.format_exc())
        if send_notify:
            send_telegram(
                f"❌ <b>Критическая ошибка в live_signals.py</b>\n"
                f"<pre>{traceback.format_exc()[-500:]}</pre>"
            )
    finally:
        log_path = logger.flush()
        print(f"\n📝 Лог сохранён: {log_path}")


if __name__ == "__main__":
    main()
