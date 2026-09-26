"""
strategy_bounce_v2.py — v7: только шорты, RR=1.5, фильтр тикеров, без скоринга в ML
Вход:  data/processed/bounce_enriched_*.parquet (без with_target)
Выход: data/results/strategy_results_*.parquet
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.inspection import permutation_importance
import xgboost as xgb

from config import *

warnings.filterwarnings("ignore")

# Кулдаун: минимум дней между сделками по одному тикеру
COOLDOWN_DAYS = 3
# Минимальный скор для входа (0 = торгуем все касания уровней)
MIN_SCORE = 0.0
# Тикеры-аутсайдеры (win rate < 45% на шортах или отрицательный PnL)
BAD_TICKERS = {"GMKN", "NVTK", "SBER", "CHMF", "PLZL", "TATN", "VTBR", "YDEX"}


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
def load_data():
    files = sorted(glob.glob(
        os.path.join(DATA_PROCESSED, "bounce_enriched_*.parquet")
    ))
    if not files:
        raise FileNotFoundError("Нет enriched данных в data/processed/")
    files = [f for f in files if "with_target" not in f]
    if not files:
        files = sorted(glob.glob(
            os.path.join(DATA_PROCESSED, "bounce_enriched_*.parquet")
        ))
    print(f"  Загружаем: {os.path.basename(files[-1])}")
    df = pd.read_parquet(files[-1])
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Симуляция сделки
# ---------------------------------------------------------------------------
def simulate_trade(df, entry_idx, direction, atr_value):
    entry_price = df["close"].iloc[entry_idx]
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
    end_idx = min(entry_idx + MAX_HOLDING_DAYS, len(df) - 1)

    for i in range(entry_idx + 1, end_idx + 1):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]
        atr_i = df["atr_14"].iloc[i] if not pd.isna(df["atr_14"].iloc[i]) else atr_value

        if direction == "long" and low <= current_stop:
            return i, current_stop, -(abs(current_stop - entry_price) / entry_price) - TX_COST, "stop", i - entry_idx
        if direction == "short" and high >= current_stop:
            return i, current_stop, -(abs(entry_price - current_stop) / entry_price) - TX_COST, "stop", i - entry_idx

        if direction == "long" and high >= take_price:
            return i, take_price, (take_price - entry_price) / entry_price - TX_COST, "take", i - entry_idx
        if direction == "short" and low <= take_price:
            return i, take_price, (entry_price - take_price) / entry_price - TX_COST, "take", i - entry_idx

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

    last_close = df["close"].iloc[end_idx]
    if direction == "long":
        pnl = (last_close - entry_price) / entry_price - TX_COST
    else:
        pnl = (entry_price - last_close) / entry_price - TX_COST
    return end_idx, last_close, pnl, "timeout", end_idx - entry_idx


# ---------------------------------------------------------------------------
# Сбор сделок
# ---------------------------------------------------------------------------
def collect_trades(df):
    all_trades = []
    directions_cfg = DIRECTIONS if isinstance(DIRECTIONS, list) else ["long", "short"]

    for ticker, grp in df.groupby("ticker"):
        if ticker in BAD_TICKERS:
            continue

        grp = grp.reset_index(drop=True)
        n = len(grp)
        last_entry_idx = -999

        for i in range(n):
            if i - last_entry_idx < COOLDOWN_DAYS:
                continue
            if pd.isna(grp["atr_14"].iloc[i]) or grp["atr_14"].iloc[i] == 0:
                continue

            score_long = grp["bounce_score_long"].iloc[i] if "bounce_score_long" in grp.columns else 0
            score_short = grp["bounce_score_short"].iloc[i] if "bounce_score_short" in grp.columns else 0

            direction = None
            score = 0.0

            if "long" in directions_cfg and score_long > MIN_SCORE and score_long >= score_short:
                direction = "long"
                score = score_long
            elif "short" in directions_cfg and score_short > MIN_SCORE and score_short > score_long:
                direction = "short"
                score = score_short

            if direction is None:
                continue

            # Трендовый фильтр
            if USE_TREND_FILTER and "ma_200" in grp.columns:
                ma200 = grp["ma_200"].iloc[i]
                close_i = grp["close"].iloc[i]
                if not pd.isna(ma200):
                    if direction == "long" and close_i < ma200:
                        continue
                    if direction == "short" and close_i > ma200:
                        continue

            exit_i, exit_price, pnl, reason, hold = simulate_trade(
                grp, i, direction, grp["atr_14"].iloc[i]
            )

            trade = {
                "entry_date": grp["date"].iloc[i],
                "exit_date": grp["date"].iloc[exit_i],
                "ticker": ticker,
                "direction": direction,
                "entry_price": grp["close"].iloc[i],
                "exit_price": exit_price,
                "pnl_pct": pnl,
                "exit_reason": reason,
                "holding_days": hold,
                "atr_at_entry": grp["atr_14"].iloc[i],
                "bounce_quality_score": score,
                "rsi_14": grp["rsi_14"].iloc[i],
                "vol_ratio": grp["vol_ratio"].iloc[i] if not pd.isna(grp["vol_ratio"].iloc[i]) else 1.0,
                "atr_14": grp["atr_14"].iloc[i],
                "ma_50": grp["ma_50"].iloc[i] if not pd.isna(grp["ma_50"].iloc[i]) else 0.0,
                "ma_200": grp["ma_200"].iloc[i] if not pd.isna(grp["ma_200"].iloc[i]) else 0.0,
                "close_to_ma50": abs(grp["close"].iloc[i] - grp["ma_50"].iloc[i]) / grp["atr_14"].iloc[i]
                    if not pd.isna(grp["ma_50"].iloc[i]) else 0.0,
            }
            all_trades.append(trade)
            last_entry_idx = exit_i

    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) > 0:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    return trades_df


# ---------------------------------------------------------------------------
# Валидация скоринга
# ---------------------------------------------------------------------------
def validate_scoring(trades_df):
    results = {}
    df = trades_df.dropna(subset=["bounce_quality_score", "pnl_pct"]).reset_index(drop=True)
    df["target_binary"] = (df["pnl_pct"] > 0).astype(int)

    rho, pval = spearmanr(df["bounce_quality_score"], df["pnl_pct"])
    results["spearman_rho"] = rho
    results["spearman_pval"] = pval

    n_bins = min(5, len(df) // 10)
    n_bins = max(n_bins, 2)
    df["score_bin"] = pd.qcut(df["bounce_quality_score"], q=n_bins, labels=False, duplicates="drop")
    bin_stats = df.groupby("score_bin").agg(
        count=("target_binary", "count"),
        win_rate=("target_binary", "mean"),
        avg_return=("pnl_pct", "mean"),
        avg_score=("bounce_quality_score", "mean"),
    ).round(4)
    results["bin_stats"] = bin_stats

    def _calc_iv(data, feature, target, bins=10):
        d = data.copy()
        d["bin"] = pd.qcut(d[feature], q=bins, labels=False, duplicates="drop")
        g = d.groupby("bin")[target].agg(["count", "sum"])
        g["non_event"] = g["count"] - g["sum"]
        g["pct_event"] = g["sum"] / g["sum"].sum()
        g["pct_non_event"] = g["non_event"] / g["non_event"].sum()
        g["woe"] = np.log((g["pct_event"] + 1e-8) / (g["pct_non_event"] + 1e-8))
        g["iv_comp"] = (g["pct_event"] - g["pct_non_event"]) * g["woe"]
        return g["iv_comp"].sum()

    iv = _calc_iv(df, "bounce_quality_score", "target_binary")
    results["iv"] = iv

    tscv = TimeSeriesSplit(n_splits=min(5, len(df) // 30))
    iv_per_fold = []
    for train_idx, test_idx in tscv.split(df):
        test_data = df.iloc[test_idx]
        if len(test_data) < 20:
            continue
        iv_f = _calc_iv(test_data, "bounce_quality_score", "target_binary")
        iv_per_fold.append(iv_f)

    results["iv_mean"] = np.mean(iv_per_fold) if iv_per_fold else 0.0
    results["iv_std"] = np.std(iv_per_fold) if iv_per_fold else 0.0
    results["iv_cv"] = (results["iv_std"] / results["iv_mean"]
                        if results["iv_mean"] != 0 else float("inf"))

    return results


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------
def make_model():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        **XGB_PARAMS,
    )


def train_walkforward(trades_df, features, n_splits=5, confidence_threshold=0.55):
    df = trades_df.dropna(subset=features + ["pnl_pct"]).reset_index(drop=True)
    df["target_binary"] = (df["pnl_pct"] > 0).astype(int)

    max_splits = max(min(n_splits, len(df) // 30), 2)
    tscv = TimeSeriesSplit(n_splits=max_splits)

    oof_proba = np.zeros(len(df))
    oof_preds = np.zeros(len(df))
    fold_metrics = []
    all_importances = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        X_train = df.iloc[train_idx][features].fillna(0.0)
        y_train = df.iloc[train_idx]["target_binary"]
        X_test = df.iloc[test_idx][features].fillna(0.0)
        y_test = df.iloc[test_idx]["target_binary"]

        if len(np.unique(y_train)) < 2:
            continue

        model = make_model()
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        oof_proba[test_idx] = proba
        oof_preds[test_idx] = (proba > confidence_threshold).astype(int)

        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = 0.5
        prec = precision_score(y_test, oof_preds[test_idx], zero_division=0)
        rec = recall_score(y_test, oof_preds[test_idx], zero_division=0)

        fold_metrics.append({
            "fold": fold, "n_train": len(train_idx), "n_test": len(test_idx),
            "precision": round(prec, 4), "recall": round(rec, 4), "auc": round(auc, 4),
        })

        try:
            perm = permutation_importance(
                model, X_test, y_test, n_repeats=5,
                random_state=42, scoring="roc_auc"
            )
            for j, feat in enumerate(features):
                all_importances.append({
                    "fold": fold, "feature": feat,
                    "importance_mean": perm.importances_mean[j],
                    "importance_std": perm.importances_std[j],
                })
        except Exception:
            pass

    if all_importances:
        imp_df = pd.DataFrame(all_importances)
        perm_summary = imp_df.groupby("feature").agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_std", "mean"),
        ).sort_values("importance_mean", ascending=False)
    else:
        perm_summary = pd.DataFrame()

    return {
        "oof_proba": oof_proba, "oof_preds": oof_preds,
        "fold_metrics": fold_metrics,
        "permutation_importance": perm_summary,
    }


# ---------------------------------------------------------------------------
# A/B бэктест
# ---------------------------------------------------------------------------
def backtest_ab(trades_df, features_with, features_without, confidence_threshold=0.55):
    df = trades_df.dropna(
        subset=list(set(features_with + features_without + ["pnl_pct"]))
    ).reset_index(drop=True)
    df["target_binary"] = (df["pnl_pct"] > 0).astype(int)

    results = {}

    for name, feats in [("with_score", features_with),
                        ("without_score", features_without)]:
        wf = train_walkforward(df, feats, n_splits=WF_N_SPLITS,
                               confidence_threshold=confidence_threshold)

        mask = wf["oof_proba"] > confidence_threshold
        selected = df.iloc[mask]

        if len(selected) > 0:
            returns = selected["pnl_pct"]
            wr = (returns > 0).mean()
            avg_ret = returns.mean()
            std = returns.std()
            sharpe = (avg_ret / std * np.sqrt(252)) if std > 0 else 0.0
            cum = returns.cumsum()
            max_dd = (cum.cummax() - cum).max() if len(cum) > 0 else 0.0
        else:
            wr = avg_ret = sharpe = max_dd = 0.0

        results[name] = {
            "n_trades": int(mask.sum()),
            "win_rate": round(wr, 4),
            "avg_return": round(avg_ret, 4),
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 4),
        }

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# Эквити-кривая
# ---------------------------------------------------------------------------
def print_equity_curve(trades_df):
    pnl = trades_df["pnl_pct"].values
    cum = np.cumsum(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total_pnl = cum[-1]
    wr = (pnl > 0).mean()
    if len(losses) > 0 and len(wins) > 0:
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        pf = (wins.sum()) / (abs(losses.sum())) if losses.sum() != 0 else float("inf")
    else:
        avg_win = avg_loss = pf = 0.0
    max_dd = (np.maximum.accumulate(cum) - cum).max()

    print(f"   Total PnL:        {total_pnl:+.4f}")
    print(f"   Win rate:         {wr:.1%}")
    print(f"   Profit factor:    {pf:.2f}")
    print(f"   Avg win:          {avg_win:+.4f}")
    print(f"   Avg loss:         {avg_loss:+.4f}")
    print(f"   Max drawdown:     {max_dd:.4f}")
    print(f"   Trades:           {len(pnl)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("strategy_bounce_v2.py v7 — шорты, RR=1.5, фильтр тикеров")
    print(f"Порог скора: {MIN_SCORE}, Кулдаун: {COOLDOWN_DAYS} дней")
    print(f"Стоп: {ATR_MULT_STOP}x ATR, Тейк: {RR_RATIO}R")
    print(f"Трейл: после {TRAIL_AFTER_RR}R, Max hold: {MAX_HOLDING_DAYS} дней")
    print(f"Направления: {DIRECTIONS}")
    print(f"Трендовый фильтр: {USE_TREND_FILTER}")
    print(f"Исключённые тикеры: {BAD_TICKERS if BAD_TICKERS else 'нет'}")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет: {len(df)} строк, {df['ticker'].nunique()} тикеров")
    print(f"Даты: {df['date'].min().date()} — {df['date'].max().date()}")

    # --- Сбор сделок ---
    print("\n1. Сбор сигналов и симуляция сделок...")
    trades = collect_trades(df)
    print(f"   Сделок: {len(trades)}")

    if len(trades) < MIN_TRADES:
        print(f"   ⚠️  Мало сделок ({len(trades)} < {MIN_TRADES}).")
        print("   Снизьте MIN_SCORE или COOLDOWN_DAYS.")
        return

    long_n = (trades["direction"] == "long").sum()
    short_n = (trades["direction"] == "short").sum()
    print(f"   Long: {long_n}, Short: {short_n}")

    print(f"\n   --- Общая статистика ---")
    print_equity_curve(trades)

    print(f"\n   Exit reasons: {dict(trades['exit_reason'].value_counts())}")

    # --- Валидация ---
    print("\n" + "=" * 60)
    print("2. ВАЛИДАЦИЯ СКОРИНГА")
    print("=" * 60)
    val = validate_scoring(trades)
    print(f"   Spearman rho:      {val['spearman_rho']:.4f} (p={val['spearman_pval']:.4f})")
    print(f"   IV:                {val['iv']:.4f}")
    print(f"   Walk-Forward IV:   {val['iv_mean']:.4f} +/- {val['iv_std']:.4f} (CV={val['iv_cv']:.2f})")
    print(f"\n   Биннинг:")
    print(val["bin_stats"].to_string())

    # --- Walk-Forward ---
    features_with = FEATURES_WITH_SCORE
    features_without = FEATURES_WITHOUT_SCORE

    print("\n" + "=" * 60)
    print("3. WALK-FORWARD ОБУЧЕНИЕ")
    print("=" * 60)
    wf = train_walkforward(trades, features_with, n_splits=WF_N_SPLITS,
                           confidence_threshold=CONFIDENCE_THRESHOLD)
    print(f"\n   Метрики по фолдам:")
    for fm in wf["fold_metrics"]:
        print(f"     Fold {fm['fold']}: prec={fm['precision']:.4f}  "
              f"rec={fm['recall']:.4f}  auc={fm['auc']:.4f}  "
              f"(train={fm['n_train']}, test={fm['n_test']})")

    print(f"\n   Permutation Importance:")
    if len(wf["permutation_importance"]) > 0:
        print(wf["permutation_importance"].head(10).to_string())
    else:
        print("   (недостаточно данных)")

    # --- A/B ---
    print("\n" + "=" * 60)
    print("4. A/B БЭКТЕСТ (ML-фильтр vs все сделки)")
    print("=" * 60)
    ab = backtest_ab(trades, features_with, features_without,
                     confidence_threshold=CONFIDENCE_THRESHOLD)
    print()
    print(ab.to_string())

    wr_delta = ab.loc["with_score", "win_rate"] - ab.loc["without_score", "win_rate"]
    sharpe_delta = ab.loc["with_score", "sharpe"] - ab.loc["without_score", "sharpe"]
    print(f"\n   Win rate delta:    {wr_delta:+.4f}")
    print(f"   Sharpe delta:      {sharpe_delta:+.4f}")
    if wr_delta > 0 and sharpe_delta > 0:
        print("   ✅ ML-фильтр улучшает стратегию")
    else:
        print("   ❌ ML-фильтр не улучшает стратегию")

    # --- Полная стратегия ---
    print("\n" + "=" * 60)
    print("5. ПОЛНАЯ СТРАТЕГИЯ (все сделки, без ML-фильтра)")
    print("=" * 60)
    print_equity_curve(trades)

    # --- Разбивка PnL ---
    print("\n" + "=" * 60)
    print("6. РАЗБИВКА PnL")
    print("=" * 60)

    if "direction" in trades.columns and trades["direction"].nunique() > 1:
        print("\n   По направлению:")
        dir_stats = trades.groupby("direction").agg(
            count=("pnl_pct", "count"),
            sum=("pnl_pct", "sum"),
            mean=("pnl_pct", "mean"),
            win_rate=("pnl_pct", lambda x: (x > 0).mean()),
        ).round(4)
        print(dir_stats.to_string())

    print("\n   По тикерам:")
    tick_stats = trades.groupby("ticker").agg(
        count=("pnl_pct", "count"),
        sum=("pnl_pct", "sum"),
        mean=("pnl_pct", "mean"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    ).round(4).sort_values("sum", ascending=False)
    print(tick_stats.to_string())

    print("\n   По причине выхода:")
    reason_stats = trades.groupby("exit_reason").agg(
        count=("pnl_pct", "count"),
        sum=("pnl_pct", "sum"),
        mean=("pnl_pct", "mean"),
    ).round(4)
    print(reason_stats.to_string())

    # --- Сохранение ---
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(DATA_RESULTS, f"strategy_results_{ts}.parquet")
    trades.to_parquet(out, index=False)
    print(f"\nСохранено: {out}")
    print("\nГотово!")


if __name__ == "__main__":
    main()
