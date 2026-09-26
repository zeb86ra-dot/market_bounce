"""
prepare_features.py — Подготовка признаков с bounce-стратегией
"""

import os
import warnings
import numpy as np
import pandas as pd
from config import *

warnings.filterwarnings("ignore")


def load_raw_candles():
    dfs = []
    for ticker in TICKERS:
        fpath = os.path.join(DATA_LIVE, f"{ticker}.csv")
        if not os.path.exists(fpath):
            print(f"  [!] Нет данных для {ticker}")
            continue
        df = pd.read_csv(fpath)
        df.columns = [c.strip().lower() for c in df.columns]
        df["Date"] = pd.to_datetime(df.get("date", df.get("tradedate")), errors="coerce")
        df = df.dropna(subset=["Date"])
        for c in ["open","high","low","close","volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("Date").reset_index(drop=True)
        df["Ticker"] = ticker
        dfs.append(df)
        print(f"  {ticker}: {len(df)} строк")
    if not dfs:
        raise FileNotFoundError("Нет данных в data/live/")
    return pd.concat(dfs, ignore_index=True)


def compute_technical(df):
    close = df["close"]; high = df["high"]; low = df["low"]; open_ = df["open"]
    volume = df["volume"]

    ma20 = close.rolling(MA_PERIOD, min_periods=5).mean()
    ma50 = close.rolling(MA_LONG, min_periods=10).mean()
    df["ma_ratio"] = close/ma20 - 1
    df["ma_trend"] = (ma20 > ma20.shift(5)).astype(int)
    df["price_above_ma"] = (close > ma20).astype(int)
    df["close_to_ma20"] = (close - ma20).abs()/ma20

    delta = close.diff()
    gain = delta.where(delta>0,0).rolling(RSI_PERIOD,min_periods=5).mean()
    loss = (-delta.where(delta<0,0)).rolling(RSI_PERIOD,min_periods=5).mean()
    rs = gain/loss.replace(0,1e-10)
    rsi = 100 - 100/(1+rs)
    df["rsi"] = rsi
    df["rsi_oversold"] = (rsi<RSI_OVERSOLD).astype(int)
    df["rsi_overbought"] = (rsi>RSI_OVERBOUGHT).astype(int)
    df["rsi_ma"] = rsi/rsi.rolling(10,min_periods=3).mean().replace(0, np.nan)

    ema12 = close.ewm(span=12,adjust=False).mean()
    ema26 = close.ewm(span=26,adjust=False).mean()
    macd = ema12-ema26
    msig = macd.ewm(span=9,adjust=False).mean()
    df["macd"] = macd; df["macd_signal"] = msig; df["macd_hist"] = macd-msig

    bb_mid = close.rolling(BB_PERIOD,min_periods=5).mean()
    bb_std = close.rolling(BB_PERIOD,min_periods=5).std()
    bb_up = bb_mid + BB_STD*bb_std
    bb_lo = bb_mid - BB_STD*bb_std
    bb_rng = (bb_up - bb_lo).replace(0, np.nan)
    df["bb_width"] = (bb_up-bb_lo)/bb_mid.replace(0, np.nan)
    df["bb_pctb"] = (close-bb_lo)/bb_rng
    df["dist_to_lower_bb"] = (close-bb_lo)/close
    df["dist_to_upper_bb"] = (bb_up-close)/close
    bb_wma = df["bb_width"].rolling(20,min_periods=5).mean()
    df["bb_squeeze"] = (df["bb_width"]<bb_wma).astype(int)

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["momentum"] = close/close.shift(10)-1

    rmax = close.rolling(20,min_periods=5).max()
    df["drawdown"] = close/rmax-1
    df["max_dd_20d"] = rmax/close-1

    vma20 = volume.rolling(20,min_periods=5).mean()
    vma5 = volume.rolling(5,min_periods=2).mean()
    df["vol_ratio"] = volume/vma20.replace(0, np.nan)
    df["vol_decline"] = (volume<vma5).astype(int)
    df["vol_trend"] = vma5/vma20.replace(0, np.nan)

    df["vol_20d"] = df["ret_1d"].rolling(20,min_periods=5).std()
    df["vol_skew"] = df["ret_1d"].rolling(20,min_periods=5).skew()

    dr = (high-low).replace(0, np.nan)
    df["high_low_range"] = dr/close
    df["close_position"] = (close-low)/dr
    df["body_ratio"] = (close-open_).abs()/dr
    df["gap"] = open_/close.shift(1)-1
    return df


def compute_bounce_features(df):
    close = df["close"]; high = df["high"]; low = df["low"]
    lo5 = low.rolling(5,min_periods=3).min()
    hi5 = high.rolling(5,min_periods=3).max()
    df["dist_to_5d_low"] = (close-lo5)/close
    df["dist_to_5d_high"] = (hi5-close)/close
    df["near_support"] = (df["dist_to_5d_low"]<NEAR_LEVEL_THRESHOLD).astype(int)
    df["near_resistance"] = (df["dist_to_5d_high"]<NEAR_LEVEL_THRESHOLD).astype(int)
    df["pullback_depth"] = df["dist_to_5d_high"]
    ma50 = close.rolling(MA_LONG,min_periods=10).mean()
    df["pullback_uptrend"] = ((close>ma50) & (close<close.rolling(5,min_periods=3).max()*0.98)).astype(int)
    df["bounce_long_signal"] = ((df["rsi"]<RSI_OVERSOLD) & (df["dist_to_5d_low"]<NEAR_LEVEL_THRESHOLD)).astype(int)
    df["bounce_short_signal"] = ((df["rsi"]>RSI_OVERBOUGHT) & (df["dist_to_5d_high"]<NEAR_LEVEL_THRESHOLD)).astype(int)
    return df


def compute_market_features(df):
    df["breadth"] = df.groupby("Date")["ret_1d"].transform(lambda x:(x>0).mean())
    df["sector_rank"] = df.groupby("Date")["ret_1d"].rank(method="average", pct=True)
    return df


def compute_regimes(df):
    ma50 = df.groupby("Ticker")["close"].transform(lambda x: x.rolling(MA_LONG,min_periods=10).mean())
    df["regime"] = "NEUTRAL"
    df.loc[(df["close"]>ma50)&(df["breadth"]>0.5),"regime"] = "BULL"
    df.loc[(df["close"]<ma50)&(df["breadth"]<0.5),"regime"] = "BEAR"
    v20 = df.groupby("Ticker")["ret_1d"].transform(lambda x: x.rolling(20,min_periods=5).std())
    vth = df.groupby("Ticker")["vol_20d"].transform(lambda x: x.quantile(VOL_PERCENTILE/100))
    df["vol_regime"] = "calm"
    df.loc[v20>vth,"vol_regime"] = "turbulent"
    return df


def compute_targets(df):
    df["target_5d"] = df.groupby("Ticker")["close"].transform(lambda x: x.shift(-HORIZON)/x-1)
    df["y_bin"] = (df["target_5d"]>0).astype(int)
    df["y_bin_rel"] = df.groupby("Date")["target_5d"].transform(lambda x:(x>x.median()).astype(int))
    return df.dropna(subset=["target_5d"]).copy()


def main():
    print("="*60)
    print("prepare_features.py")
    print("="*60)

    print("\n[1] Загрузка свечей...")
    candles = load_raw_candles()
    print(f"  Всего: {len(candles)} строк")

    print("\n[2] Технические индикаторы и bounce-признаки...")
    processed = []
    for t in candles["Ticker"].unique():
        dt = candles[candles["Ticker"]==t].copy().sort_values("Date").reset_index(drop=True)
        dt = compute_technical(dt)
        dt = compute_bounce_features(dt)
        processed.append(dt)
    df = pd.concat(processed, ignore_index=True)

    for col in ["brent_ret", "usdrub_ret", "gold_ret"]:
        df[col] = 0.0

    print("\n[3] Рыночные признаки...")
    df = compute_market_features(df)

    print("\n[4] Режимы рынка...")
    df = compute_regimes(df)

    print("\n[5] Таргеты...")
    df = compute_targets(df)

    df = df.drop_duplicates(subset=["Date","Ticker"]).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [!] Отсутствуют: {missing}")
        for c in missing: df[c] = 0.0

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fpath = os.path.join(DATA_PROCESSED, f"market_ml_ready_{ts}.parquet")
    df.to_parquet(fpath, index=False)
    print(f"\nСохранено: {fpath}")
    print(f"Строк: {len(df)}, тикеров: {df['Ticker'].nunique()}")
    print(f"y_bin: {(df['y_bin']==1).mean():.1%} pos")
    print(f"Период: {df['Date'].min().date()} — {df['Date'].max().date()}")
    print("Готово!")


if __name__ == "__main__":
    main()
