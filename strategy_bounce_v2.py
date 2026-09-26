"""
strategy_bounce_v2.py — bounce-стратегия v2
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

warnings.filterwarnings("ignore")


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_PROCESSED, "market_ml_ready_*.parquet")))
    if not files:
        raise FileNotFoundError("Нет данных. Запустите prepare_features.py")
    df = pd.read_parquet(files[-1])
    print(f"Загружен: {os.path.basename(files[-1])}")
    print(f"  Тикеров: {df['Ticker'].nunique()}, строк: {len(df)}")
    return df


def train_models(train_df):
    models = {}
    for regime in ["BULL","BEAR","NEUTRAL"]:
        sub = train_df[train_df["regime"]==regime].dropna(subset=FEATURE_COLS+["y_bin"])
        if len(sub) < 30:
            print(f"    {regime}: пропущен ({len(sub)})")
            continue
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(sub[FEATURE_COLS], sub["y_bin"])
        models[regime] = m
        print(f"    {regime}: обучен ({len(sub)}), pos={sub['y_bin'].mean():.1%}")
    return models


def predict_proba(models, row):
    r = row.get("regime","NEUTRAL")
    if r not in models:
        r = "NEUTRAL" if "NEUTRAL" in models else list(models.keys())[0]
    X = pd.DataFrame([row[FEATURE_COLS].values], columns=FEATURE_COLS)
    return models[r].predict_proba(X)[0,1]


class Position:
    __slots__ = ["ticker","direction","entry_date","entry_price","regime","days_held","prev_close"]
    def __init__(self,ticker,direction,entry_date,entry_price,regime):
        self.ticker=ticker; self.direction=direction
        self.entry_date=entry_date; self.entry_price=entry_price
        self.regime=regime; self.days_held=0; self.prev_close=entry_price


def run_backtest(test_df, models):
    dates = sorted(test_df["Date"].unique())
    positions = []
    trades = []
    daily_returns = []
    pw = 1.0 / MAX_POSITIONS

    for date in dates:
        day = test_df[test_df["Date"]==date].set_index("Ticker")
        dpnl = 0.0
        for p in positions:
            if p.ticker not in day.index: continue
            c = day.loc[p.ticker,"close"]
            ret = c/p.prev_close-1 if p.direction=="long" else 1-c/p.prev_close
            dpnl += ret*pw
            p.prev_close = c
        daily_returns.append(dpnl)

        keep = []
        for p in positions:
            if p.ticker not in day.index:
                keep.append(p); continue
            row = day.loc[p.ticker]
            c = row["close"]
            p.days_held += 1
            reason = None
            if p.days_held >= MAX_HOLD_DAYS:
                reason = "time_stop"
            elif p.direction == "long":
                if c <= p.entry_price*(1-STOP_LOSS): reason = "stop_loss"
                elif c >= p.entry_price*(1+TAKE_PROFIT): reason = "take_profit"
                elif row["rsi"]>RSI_OVERBOUGHT or row["dist_to_5d_high"]<NEAR_LEVEL_THRESHOLD or row["dist_to_upper_bb"]<NEAR_LEVEL_THRESHOLD:
                    reason = "technical"
            if reason:
                gross = c/p.entry_price-1 if p.direction=="long" else 1-c/p.entry_price
                trades.append({
                    "ticker":p.ticker,"direction":p.direction,
                    "entry_date":p.entry_date,"exit_date":date,
                    "entry_price":p.entry_price,"exit_price":c,
                    "days_held":p.days_held,"pnl":gross-2*COMMISSION,
                    "exit_reason":reason,"regime":p.regime,
                })
            else:
                keep.append(p)
        positions = keep

        free = MAX_POSITIONS - len(positions)
        if free <= 0: continue
        for t in day.index:
            if free <= 0: break
            if any(p.ticker==t for p in positions): continue
            row = day.loc[t]
            proba = predict_proba(models, row)
            enter_long = (row["bounce_long_signal"]==1 and proba>=PROBA_LONG
                          and row["regime"]!="BEAR" and row["vol_regime"]=="calm")
            if enter_long:
                positions.append(Position(t,"long",date,row["close"],row["regime"]))
                free -= 1

    if positions:
        last = test_df[test_df["Date"]==dates[-1]].set_index("Ticker")
        for p in positions:
            if p.ticker in last.index:
                c = last.loc[p.ticker,"close"]
                gross = c/p.entry_price-1 if p.direction=="long" else 1-c/p.entry_price
                trades.append({
                    "ticker":p.ticker,"direction":p.direction,
                    "entry_date":p.entry_date,"exit_date":dates[-1],
                    "entry_price":p.entry_price,"exit_price":c,
                    "days_held":p.days_held,"pnl":gross-2*COMMISSION,
                    "exit_reason":"end_of_test","regime":p.regime,
                })

    pv = (1+pd.Series(daily_returns)).cumprod()
    return trades, daily_returns, pv


def walk_forward(df):
    dates = sorted(df["Date"].unique())
    idx = 0; all_trades = []; all_rets = []; all_pv = []; last_models = None; fold = 0
    while idx + TRAIN_WINDOW + TEST_WINDOW <= len(dates):
        tr_d = dates[idx:idx+TRAIN_WINDOW]
        te_d = dates[idx+TRAIN_WINDOW:idx+TRAIN_WINDOW+TEST_WINDOW]
        tr = df[df["Date"].isin(tr_d)]
        te = df[df["Date"].isin(te_d)]
        print(f"\n=== Fold {fold+1} ===")
        print(f"  Train: {tr_d[0].date()} — {tr_d[-1].date()}")
        print(f"  Test:  {te_d[0].date()} — {te_d[-1].date()}")
        models = train_models(tr)
        last_models = models
        t, r, p = run_backtest(te, models)
        all_trades.extend(t); all_rets.extend(r); all_pv.extend(p)
        fold += 1; idx += STEP
    if last_models:
        os.makedirs(MODELS_DIR, exist_ok=True)
        for k, m in last_models.items():
            joblib.dump(m, os.path.join(MODELS_DIR, f"model_{k}.joblib"))
            print(f"Модель {k} сохранена")
    return all_trades, all_rets, all_pv


def evaluate(trades, rets, pv):
    if not trades:
        print("Нет сделок"); return
    tdf = pd.DataFrame(trades)
    tot = tdf["pnl"].sum()
    wr = (tdf["pnl"]>0).mean()
    avg = tdf["pnl"].mean()
    dd = pd.Series(rets).cumsum().min()
    sharpe = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    print("\n"+"="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Сделок: {len(tdf)}")
    print(f"Win rate: {wr:.1%}")
    print(f"Средний PnL: {avg:.2%}")
    print(f"Суммарный PnL: {tot:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max drawdown: {dd:.2%}")

    os.makedirs(DATA_RESULTS, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    tdf.to_csv(os.path.join(DATA_RESULTS, f"trades_{ts}.csv"), index=False)
    pd.DataFrame({"daily_return":rets,"portfolio_value":pv}).to_csv(
        os.path.join(DATA_RESULTS, f"equity_{ts}.csv"), index=False)

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(pv, color="blue"); plt.title("Equity Curve"); plt.grid(True)
    plt.savefig(os.path.join(GRAPHS_DIR, f"equity_curve_{ts}.png"), dpi=100, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10,6))
    plt.hist(tdf["pnl"]*100, bins=20, color="green", edgecolor="black")
    plt.title("PnL Distribution (%)"); plt.grid(True)
    plt.savefig(os.path.join(GRAPHS_DIR, f"pnl_distribution_{ts}.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("Сохранено в data/results/ и results_graphs/")


def main():
    print("="*60)
    print("strategy_bounce_v2.py")
    print("="*60)
    df = load_data()
    t, r, p = walk_forward(df)
    evaluate(t, r, p)


if __name__ == "__main__":
    main()
