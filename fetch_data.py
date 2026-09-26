"""
fetch_data.py — загрузка дневных свечей с MOEX ISS API (с пагинацией)
"""

import os
import time
import pandas as pd
import requests
from datetime import timedelta
from pathlib import Path

TICKERS = [
    "GAZP", "LKOH", "NVTK", "ROSN", "TATN", "GMKN", "NLMK", "ALRS",
    "PLZL", "CHMF", "SBER", "VTBR", "MGNT", "YDEX",
]

FETCH_DAYS = 730

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

MOEX_STOCK_URL = (
    "https://iss.moex.com/iss/history/engines/stock/"
    "markets/shares/boards/TQBR/securities/{ticker}.json"
)


def fetch_stock(ticker, start_date, end_date):
    """Скачивает все свечи одной акции с пагинацией по 100 строк."""
    url = MOEX_STOCK_URL.format(ticker=ticker)
    all_rows = []
    start_idx = 0

    while True:
        params = {"from": start_date, "till": end_date, "start": start_idx}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        columns = data["history"]["columns"]
        rows = data["history"]["data"]
        if not rows:
            break

        all_rows.extend(rows)
        if len(rows) < 100:
            break
        start_idx += 100
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=columns)
    close_col = "CLOSE" if "CLOSE" in df.columns else ("LAST" if "LAST" in df.columns else None)
    if "TRADEDATE" not in df.columns or not close_col:
        return pd.DataFrame()

    keep = {
        "TRADEDATE": "Date",
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        close_col: "Close",
        "VOLUME": "Volume",
    }
    keep = {k: v for k, v in keep.items() if k in df.columns}
    df = df[list(keep.keys())].rename(columns=keep)
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Ticker"] = ticker
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def main():
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")

    print("=" * 60)
    print("fetch_data.py — загрузка данных с MOEX")
    print(f"Период: {start} — {end} ({FETCH_DAYS} дней)")
    print(f"Тикеров: {len(TICKERS)}")
    print("=" * 60)

    all_stock = []
    for i, t in enumerate(TICKERS):
        print(f"  [{i+1}/{len(TICKERS)}] {t}...", end=" ", flush=True)
        try:
            df = fetch_stock(t, start, end)
            if len(df) > 0:
                all_stock.append(df)
                print(f"OK ({len(df)} строк)")
            else:
                print("нет данных")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)

    if not all_stock:
        print("❌ Не удалось загрузить ни одного тикера.")
        return

    full = pd.concat(all_stock, ignore_index=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_RAW / f"raw_candles_{ts}.parquet"
    full.to_parquet(out_path, index=False)

    print(f"\nАкции сохранены: {out_path}")
    print(f"Всего строк: {len(full)}")
    print(f"Тикеров: {full['Ticker'].nunique()}")
    print(f"Даты: {full['Date'].min().date()} — {full['Date'].max().date()}")
    print("\nГотово!")


if __name__ == "__main__":
    main()
