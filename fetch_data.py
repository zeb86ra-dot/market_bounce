"""
fetch_data.py — загрузка дневных свечей с MOEX ISS API
Сохраняет CSV в data/live/ для каждого тикера из config.TICKERS
"""

import os
import time
import requests
import pandas as pd
import apimoex
from config import TICKERS, DATA_LIVE


def fetch_ticker_history(ticker, days=730):
    session = requests.Session()
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)
    try:
        data = apimoex.get_board_history(
            session, security=ticker, market='shares', board='TQBR',
            columns=['TRADEDATE','OPEN','HIGH','LOW','CLOSE','VOLUME'],
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
        )
        df = pd.DataFrame(data)
        if df.empty:
            print(f"  Пустой ответ для {ticker}")
            return None
        df['Date'] = pd.to_datetime(df['TRADEDATE'])
        df = df[['Date','OPEN','HIGH','LOW','CLOSE','VOLUME']].copy()
        df.columns = ['Date','Open','High','Low','Close','Volume']
        return df.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        print(f"  Ошибка: {e}")
        return None


def main():
    os.makedirs(DATA_LIVE, exist_ok=True)
    print("="*60)
    print("fetch_data.py — загрузка данных с MOEX")
    print(f"Тикеров: {len(TICKERS)}")
    print("="*60)
    for i, ticker in enumerate(TICKERS, 1):
        print(f"  [{i}/{len(TICKERS)}] {ticker}...", end=" ")
        df = fetch_ticker_history(ticker)
        if df is not None:
            fname = os.path.join(DATA_LIVE, f"{ticker}.csv")
            df.to_csv(fname, index=False)
            print(f"OK ({len(df)} строк)")
        else:
            print("FAIL")
        time.sleep(0.3)
    print("\nГотово!")


if __name__ == "__main__":
    main()
