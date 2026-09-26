"""
prepare_features.py — Подготовка признаков с bounce-стратегией
Выход: parquet-файл с 44 признаками + таргеты + режимы
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from config import *

warnings.filterwarnings("ignore")


def load_raw_candles():
    """Загрузка сырых свечей из data/live/"""
    dfs = []
    for ticker in TICKERS:
        patterns = [
            os.path.join(DATA_LIVE, f"{ticker}.csv"),
            os.path.join(DATA_LIVE, f"{ticker}_*.csv"),
            os.path.join(DATA_LIVE, f"*{ticker}*.csv"),
        ]
        found = []
        for p in patterns:
            found.extend(glob.glob(p))
        if not found:
            print(f"  [!] Нет данных для {ticker}")
            continue
        fpath = found[0]
        df = pd.read_csv(fpath)
        df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]
        col_map = {}
        for target, candidates in {
            "date": ["date", "datetime", "day", "trade_date"],
            "open": ["open", "openprice"],
            "high": ["high", "highprice"],
            "low": ["low", "lowprice"],
            "close": ["close", "closeprice", "last"],
            "volume": ["vol", "volume", "value"],
        }.items():
            for cand in candidates:
                if cand in df.columns:
                    col_map[cand] = target
                    break
        df = df.rename(columns=col_map)
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df["Ticker"] = ticker
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("Date").reset_index(drop=True)
        dfs.append(df)
        print(f"  {ticker}: {len(df)} строк, {df['Date'].min().date()} — {df['Date'].max().date()}")
    if not dfs:
        raise FileNotFoundError("Не найдены файлы свечей в data/live/")
    return pd.concat(dfs, ignore_index=True)


def load_macro():
    """Загрузка макро-данных (Brent, USDRUB, Gold)"""
    macro = pd.DataFrame()
    macro_path = os.path.join(DATA_LIVE, "macro.csv")
    if os.path.exists(macro_path):
        macro = pd.read_csv(macro_path)
        macro.columns = [c.strip().lower() for c in macro.columns]
        date_col = "date" if "date" in macro.columns else macro.columns[0]
        macro["Date"] = pd.to_datetime(macro[date_col], errors="coerce")
        macro = macro.dropna(subset=["Date"])
        for col in ["brent", "usdrub", "gold"]:
            if col not in macro.columns:
                for c in macro.columns:
                    if col in c.lower():
                        macro[col] = pd.to_numeric(macro[c], errors="coerce")
                        break
            else:
                macro[col] = pd.to_numeric(macro[col], errors="coerce")
        return macro[["Date", "brent", "usdrub", "gold"]].copy()

    for name, fname in [("brent", "Brent.csv"), ("usdrub", "USDRUB.csv"), ("gold", "Gold.csv")]:
        fpath = os.path.join(DATA_LIVE, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(DATA_LIVE, fname.lower())
        if os.path.exists(fpath):
            tmp = pd.read_csv(fpath)
            tmp.columns = [c.strip().lower() for c in tmp.columns]
            date_col = "date" if "date" in tmp.columns else tmp.columns[0]
            tmp["Date"] = pd.to_datetime(tmp[date_col], errors="coerce")
            price_col = "close" if "close" in tmp.columns else tmp.columns[1]
            tmp[name] = pd.to_numeric(tmp[price_col], errors="coerce")
            tmp = tmp[["Date", name]].dropna(subset=["Date"])
            if macro.empty:
                macro = tmp
            else:
                macro = macro.merge(tmp, on="Date", how="outer")
    if macro.empty:
        print("  [!] Макро-данные не найдены, будут заполнены нулями")
        return None
    return macro.sort_values("Date").reset_index(drop=True)


def compute_technical(df):
    """Технические индикаторы"""
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    open_ = df["open"] if "open" in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    ma20 = close.rolling(MA_PERIOD, min_periods=5).mean()
    ma50 = close.rolling(MA_LONG, min_periods=10).mean()
    df["ma_ratio"] = close / ma20 - 1
    df["ma_trend"] = (ma20 > ma20.shift(5)).astype(int)
    df["price_above_ma"] = (close > ma20).astype(int)
    df["close_to_ma20"] = (close - ma20).abs() / ma20

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(RSI_PERIOD, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(RSI_PERIOD, min_periods=5).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    df["rsi"] = rsi
    df["rsi_oversold"] = (rsi < RSI_OVERSOLD).astype(int)
    df["rsi_overbought"] = (rsi > RSI_OVERBOUGHT).astype(int)
    df["rsi_ma"] = rsi / rsi.rolling(10, min_periods=3).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    bb_mid = close.rolling(BB_PERIOD, min_periods=5).mean()
    bb_std = close.rolling(BB_PERIOD, min_periods=5).std()
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, 1e-10)
    bb_range = (bb_upper - bb_lower).replace(0, 1e-10)
    df["bb_pctb"] = (close - bb_lower) / bb_range
    df["dist_to_lower_bb"] = (close - bb_lower) / close
    df["dist_to_upper_bb"] = (bb_upper - close) / close
    bb_width_ma = df["bb_width"].rolling(20, min_periods=5).mean()
    df["bb_squeeze"] = (df["bb_width"] < bb_width_ma).astype(int)

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["momentum"] = close / close.shift(10) - 1

    rolling_max = close.rolling(20, min_periods=5).max()
    df["drawdown"] = close / rolling_max - 1
    df["max_dd_20d"] = rolling_max / close - 1

    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ma5 = volume.rolling(5, min_periods=2).mean()
    df["vol_ratio"] = volume / vol_ma20.replace(0, 1)
    df["vol_decline"] = (volume < vol_ma5).astype(int)
    df["vol_trend"] = vol_ma5 / vol_ma20.replace(0, 1)

    df["vol_20d"] = df["ret_1d"].rolling(20, min_periods=5).std()
    df["vol_skew"] = df["ret_1d"].rolling(20, min_periods=5).skew()

    day_range = (high - low).replace(0, 1e-10)
    df["high_low_range"] = day_range / close
    df["close_position"] = (close - low) / day_range
    body = (close - open_).abs()
    df["body_ratio"] = body / day_range
    df["gap"] = open_ / close.shift(1) - 1

    return df


def compute_bounce_features(df):
    """Bounce-признаки"""
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    low_5d = low.rolling(5, min_periods=3).min()
    high_5d = high.rolling(5, min_periods=3).max()

    df["dist_to_5d_low"] = (close - low_5d) / close
    df["dist_to_5d_high"] = (high_5d - close) / close

    df["near_support"] = (df["dist_to_5d_low"] < NEAR_LEVEL_THRESHOLD).astype(int)
    df["near_resistance"] = (df["dist_to_5d_high"] < NEAR_LEVEL_THRESHOLD).astype(int)

    df["pullback_depth"] = df["dist_to_5d_high"]

    ma50 = close.rolling(MA_LONG, min_periods=10).mean()
    df["pullback_uptrend"] = ((close > ma50) & (close < close.rolling(5, min_periods=3).max() * 0.98)).astype(int)

    df["bounce_long_signal"] = (
        (df["rsi"] < RSI_OVERSOLD) & (df["dist_to_5d_low"] < NEAR_LEVEL_THRESHOLD)
    ).astype(int)

    df["bounce_short_signal"] = (
        (df["rsi"] > RSI_OVERBOUGHT) & (df["dist_to_5d_high"] < NEAR_LEVEL_THRESHOLD)
    ).astype(int)

    return df


def compute_market_features(df_all):
    df_all["breadth"] = df_all.groupby("Date")["ret_1d"].transform(lambda x: (x > 0).mean())
    df_all["sector_rank"] = df_all.groupby("Date")["ret_1d"].rank(method="average", pct=True)
    return df_all


def compute_regimes(df_all):
    ma50 = df_all.groupby("Ticker")["close"].transform(lambda x: x.rolling(MA_LONG, min_periods=10).mean())
    cond_bull = (df_all["close"] > ma50) & (df_all["breadth"] > 0.5)
    cond_bear = (df_all["close"] < ma50) & (df_all["breadth"] < 0.5)
    df_all["regime"] = "NEUTRAL"
    df_all.loc[cond_bull, "regime"] = "BULL"
    df_all.loc[cond_bear, "regime"] = "BEAR"

    vol_20d = df_all.groupby("Ticker")["ret_1d"].transform(lambda x: x.rolling(20, min_periods=5).std())
    vol_threshold = df_all.groupby("Ticker")["vol_20d"].transform(lambda x: x.quantile(VOL_PERCENTILE / 100))
    df_all["vol_regime"] = "calm"
    df_all.loc[vol_20d > vol_threshold, "vol_regime"] = "turbulent"
    return df_all


def compute_targets(df_all):
    df_all["target_5d"] = df_all.groupby("Ticker")["close"].transform(lambda x: x.shift(-HORIZON) / x - 1)
    df_all["y_bin"] = (df_all["target_5d"] > 0).astype(int)
    df_all["y_bin_rel"] = df_all.groupby("Date")["target_5d"].transform(lambda x: (x > x.median()).astype(int))
    df_all = df_all.dropna(subset=["target_5d"]).copy()
    return df_all


def add_macro(df_all, macro):
    if macro is None:
        for col in ["brent_ret", "usdrub_ret", "gold_ret"]:
            df_all[col] = 0.0
        return df_all

    macro = macro.sort_values("Date")
    df_all = df_all.sort_values("Date")

    for col in ["brent", "usdrub", "gold"]:
        if col not in macro.columns:
            df_all[f"{col}_ret"] = 0.0
            continue
        macro_vals = macro[["Date", col]].dropna(subset=[col]).sort_values("Date")
        df_all = df_all.merge(macro_vals.rename(columns={col: f"{col}_price"}), on="Date", how="left")
        df_all[f"{col}_price"] = df_all[f"{col}_price"].ffill()
        df_all[f"{col}_ret"] = df_all[f"{col}_price"].pct_change()
        df_all = df_all.drop(columns=[f"{col}_price"])

    for col in ["brent_ret", "usdrub_ret", "gold_ret"]:
        df_all[col] = df_all[col].fillna(0.0)
    return df_all


def main():
    print("=" * 60)
    print("prepare_features.py — подготовка датасета")
    print("=" * 60)

    print("\n[1] Загрузка свечей...")
    candles = load_raw_candles()
    print(f"  Всего строк: {len(candles)}")

    print("\n[2] Загрузка макро-данных...")
    macro = load_macro()
    if macro is not None:
        print(f"  Макро: {len(macro)} строк")

    print("\n[3] Вычисление индикаторов и bounce-признаков...")
    processed = []
    for ticker in candles["Ticker"].unique():
        df_t = candles[candles["Ticker"] == ticker].copy()
        df_t = compute_technical(df_t)
        df_t = compute_bounce_features(df_t)
        processed.append(df_t)
    df_all = pd.concat(processed, ignore_index=True)
    print(f"  После обработки: {len(df_all)} строк")

    print("\n[4] Добавление макро-данных...")
    df_all = add_macro(df_all, macro)

    print("\n[5] Вычисление рыночных признаков...")
    df_all = compute_market_features(df_all)

    print("\n[6] Определение режимов рынка...")
    df_all = compute_regimes(df_all)
    print(f"  Regime: {dict(df_all['regime'].value_counts())}")
    print(f"  Vol regime: {dict(df_all['vol_regime'].value_counts())}")

    print("\n[7] Вычисление таргетов...")
    df_all = compute_targets(df_all)
    print(f"  y_bin: {(df_all['y_bin'] == 1).mean():.1%} pos / {(df_all['y_bin'] == 0).mean():.1%} neg")
    print(f"  Период: {df_all['Date'].min().date()} — {df_all['Date'].max().date()}")
    print(f"  Тикеров: {df_all['Ticker'].nunique()}")
    print(f"  Строк: {len(df_all)}")

    missing = [c for c in FEATURE_COLS if c not in df_all.columns]
    if missing:
        print(f"\n  [!] Отсутствуют признаки: {missing}")
        for c in missing:
            df_all[c] = 0.0

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fname = f"market_ml_ready_{timestamp}.parquet"
    fpath = os.path.join(DATA_PROCESSED, fname)
    df_all.to_parquet(fpath, index=False)
    print(f"\n[8] Сохранено: {fpath}")
    print(f"  Колонок: {len(df_all.columns)}")
    print(f"\nГотово!")


if __name__ == "__main__":
    main()