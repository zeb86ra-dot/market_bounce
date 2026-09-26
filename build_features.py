"""
build_features.py — v5: контртрендовый score_trend + только swing-уровни + MA200
Вход:  data/raw/raw_candles_*.parquet
Выход: data/processed/bounce_enriched_*.parquet
"""

import os
import glob
import numpy as np
import pandas as pd
from config import *


def load_raw():
    files = sorted(glob.glob(os.path.join(DATA_RAW, "raw_candles_*.parquet")))
    if not files:
        raise FileNotFoundError("Нет raw_candles_*.parquet в data/raw/")
    df = pd.read_parquet(files[-1])
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def add_indicators(df):
    """ATR, MA50, MA200, RSI, avg_volume, vol_ratio."""
    df = df.copy()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(ATR_PERIOD).mean()

    df["ma_50"] = df["close"].rolling(MA_FAST).mean()
    df["ma_200"] = df["close"].rolling(MA_SLOW).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["avg_volume"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    df["vol_ratio"] = df["volume"] / df["avg_volume"].replace(0, np.nan)

    return df


def detect_swing_levels(df, lookback=50, min_touches=2, tol=0.01):
    """
    Находит значимые уровни по swing high/low с 2+ касаниями.
    Возвращает списки support_levels и resistance_levels для каждой свечи.
    """
    supports = []
    resistances = []

    lows = df["low"].values
    highs = df["high"].values
    n = len(df)

    for i in range(n):
        start = max(0, i - lookback)
        window_lows = lows[start:i + 1]
        window_highs = highs[start:i + 1]

        swing_lows = set()
        swing_highs = set()

        for j in range(start + 2, min(i, n - 1)):
            if lows[j] < lows[j - 1] and lows[j] < lows[j + 1]:
                swing_lows.add(lows[j])
            if highs[j] > highs[j - 1] and highs[j] > highs[j + 1]:
                swing_highs.add(highs[j])

        valid_supports = []
        for lvl in swing_lows:
            if lvl == 0:
                continue
            touches = np.sum(
                (window_lows <= lvl * (1 + tol)) &
                (window_highs >= lvl * (1 - tol))
            )
            if touches >= min_touches:
                valid_supports.append(lvl)

        valid_resistances = []
        for lvl in swing_highs:
            if lvl == 0:
                continue
            touches = np.sum(
                (window_lows <= lvl * (1 + tol)) &
                (window_highs >= lvl * (1 - tol))
            )
            if touches >= min_touches:
                valid_resistances.append(lvl)

        supports.append(valid_supports)
        resistances.append(valid_resistances)

    df["levels_long"] = supports
    df["levels_short"] = resistances
    return df


def calc_bounce_score_for_row(df_slice, level, direction):
    """Считает bounce_quality_score для последней строки среза."""
    i = len(df_slice) - 1
    low_i = df_slice["low"].iloc[i]
    high_i = df_slice["high"].iloc[i]
    close_i = df_slice["close"].iloc[i]

    if direction == "long":
        if low_i > level * (1 + LEVEL_TOL):
            return 0.0
        if close_i < level:
            return 0.0
    else:
        if high_i < level * (1 - LEVEL_TOL):
            return 0.0
        if close_i > level:
            return 0.0

    # 1. score_close
    if direction == "long":
        body_above = (close_i - level) / (high_i - level + 1e-8)
        score_close = np.clip(body_above, 0.0, 1.0)
    else:
        body_below = (level - close_i) / (level - low_i + 1e-8)
        score_close = np.clip(body_below, 0.0, 1.0)

    # 2. score_volume
    vr = df_slice["vol_ratio"].iloc[i]
    if pd.isna(vr):
        vr = 1.0
    score_volume = np.clip((vr - 0.5) / 1.0, 0.0, 1.0)

    # 3. score_confirm
    if i >= 1:
        prev_close = df_slice["close"].iloc[i - 1]
        prev_open = df_slice["open"].iloc[i - 1]
        if direction == "long":
            score_confirm = 1.0 if prev_close > prev_open else 0.0
        else:
            score_confirm = 1.0 if prev_close < prev_open else 0.0
    else:
        score_confirm = 0.0

    # 4. score_penetration
    atr = df_slice["atr_14"].iloc[i]
    if pd.isna(atr) or atr == 0:
        score_penetration = 0.0
    else:
        if direction == "long":
            penetration = max(level - low_i, 0.0)
        else:
            penetration = max(high_i - level, 0.0)
        pen_atr = penetration / atr
        score_penetration = np.clip(1.0 - pen_atr * 2.0, 0.0, 1.0)

    # 5. score_trend — контртрендовый
    ma50 = df_slice["ma_50"].iloc[i]
    if pd.isna(ma50) or pd.isna(atr) or atr == 0:
        score_trend = 0.0
    else:
        dist_from_ma = abs(close_i - ma50) / atr
        if direction == "long":
            score_trend = np.clip(dist_from_ma / 3.0, 0.0, 1.0) if close_i < ma50 else 0.0
        else:
            score_trend = np.clip(dist_from_ma / 3.0, 0.0, 1.0) if close_i > ma50 else 0.0

    # 6. score_tests
    test_count = 0
    check_start = max(0, i - SWING_LOOKBACK)
    for j in range(check_start, i):
        if (df_slice["low"].iloc[j] <= level * (1 + LEVEL_TOL) and
                df_slice["high"].iloc[j] >= level * (1 - LEVEL_TOL)):
            test_count += 1

    if test_count in [1, 2, 3]:
        score_tests = 1.0
    elif test_count in [4, 5]:
        score_tests = 0.5
    else:
        score_tests = 0.0

    # 7. score_rsi
    rsi = df_slice["rsi_14"].iloc[i]
    if pd.isna(rsi):
        score_rsi = 0.0
    else:
        if direction == "long":
            score_rsi = np.clip((50.0 - rsi) / 20.0, 0.0, 1.0)
        else:
            score_rsi = np.clip((rsi - 50.0) / 20.0, 0.0, 1.0)

    total = (
        score_close * SCORE_WEIGHTS["score_close"] +
        score_volume * SCORE_WEIGHTS["score_volume"] +
        score_confirm * SCORE_WEIGHTS["score_confirm"] +
        score_penetration * SCORE_WEIGHTS["score_penetration"] +
        score_trend * SCORE_WEIGHTS["score_trend"] +
        score_tests * SCORE_WEIGHTS["score_tests"] +
        score_rsi * SCORE_WEIGHTS["score_rsi"]
    )
    return float(total)


def process_ticker(df_ticker):
    """Обрабатывает один тикер: индикаторы + уровни + скоринг."""
    df_ticker = df_ticker.reset_index(drop=True)
    df_ticker = add_indicators(df_ticker)
    df_ticker = detect_swing_levels(df_ticker, lookback=50, min_touches=2, tol=0.01)

    scores_long = []
    scores_short = []

    for i in range(len(df_ticker)):
        row = df_ticker.iloc[i]

        best_s_l = 0.0
        for lvl in row["levels_long"]:
            s = calc_bounce_score_for_row(df_ticker.iloc[:i + 1], lvl, "long")
            if s > best_s_l:
                best_s_l = s
        scores_long.append(best_s_l)

        best_s_s = 0.0
        for lvl in row["levels_short"]:
            s = calc_bounce_score_for_row(df_ticker.iloc[:i + 1], lvl, "short")
            if s > best_s_s:
                best_s_s = s
        scores_short.append(best_s_s)

    df_ticker["bounce_score_long"] = scores_long
    df_ticker["bounce_score_short"] = scores_short

    return df_ticker


def main():
    print("=" * 60)
    print("build_features.py v5 — контртренд + swing-уровни + MA200")
    print("=" * 60)

    df = load_raw()
    print(f"Загружено: {len(df)} строк, {df['ticker'].nunique()} тикеров")

    tickers = df["ticker"].unique()
    all_parts = []

    for j, ticker in enumerate(tickers):
        print(f"  [{j + 1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        df_t = df[df["ticker"] == ticker].copy()
        df_t = process_ticker(df_t)
        sig_long = (df_t["bounce_score_long"] > 0).sum()
        sig_short = (df_t["bounce_score_short"] > 0).sum()
        print(f"L={sig_long} S={sig_short}")
        all_parts.append(df_t)

    df_processed = pd.concat(all_parts, ignore_index=True)
    df_processed = df_processed.drop(columns=["levels_long", "levels_short"], errors="ignore")

    if "ticker" not in df_processed.columns:
        raise RuntimeError("Колонка 'ticker' потеряна!")

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_PROCESSED, f"bounce_enriched_{ts}.parquet")
    df_processed.to_parquet(out_path, index=False)

    print(f"\n✅ Сохранён: {out_path}")
    sig_total = ((df_processed["bounce_score_long"] > 0) |
                 (df_processed["bounce_score_short"] > 0)).sum()
    print(f"Сигналов: {sig_total} ({sig_total / len(df_processed) * 100:.1f}%)")
    print("Готово!")


if __name__ == "__main__":
    main()
