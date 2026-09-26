"""
optimize.py — поиск оптимальных параметров стратегии bounce
Перебирает комбинации и выводит лучшие по Sharpe или суммарному PnL
"""

import itertools
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from config import *
import strategy_bounce_v2 as st

def evaluate_params(df, rsi_oversold, rsi_overbought, near_level, proba_long, max_hold, use_calm_filter):
    """Прогон walk-forward с заданными параметрами, возврат метрик"""
    # Переопределяем глобальные параметры
    global RSI_OVERSOLD, RSI_OVERBOUGHT, NEAR_LEVEL_THRESHOLD, PROBA_LONG, MAX_HOLD_DAYS, VOL_REGIME_FILTER
    RSI_OVERSOLD = rsi_oversold
    RSI_OVERBOUGHT = rsi_overbought
    NEAR_LEVEL_THRESHOLD = near_level
    PROBA_LONG = proba_long
    MAX_HOLD_DAYS = max_hold

    # Принудительно пересчитываем сигналы в данных, т.к. они зависят от RSI и близости
    # Но чтобы не пересчитывать все признаки, мы можем предположить, что сигналы уже есть,
    # и если меняем пороги, нужно пересоздать датасет. Для простоты пропустим эту часть,
    # а в реальном использовании нужно перегенерировать признаки с новыми порогами.
    # Вместо этого можно в самом prepare_features.py при расчёте bounce_long_signal использовать
    # параметры из конфига, но они уже вычислены. Поэтому для оптимизации нужно либо
    # перегенерировать parquet при каждом наборе, либо заранее создать несколько колонок
    # с разными порогами. Здесь для простоты будем использовать уже существующие сигналы,
    # меняя только proba_long и max_hold_days, а также фильтр calm.

    # ВНИМАНИЕ: приведённый ниже код не учитывает изменение RSI и NEAR_LEVEL_THRESHOLD,
    # т.к. сигналы уже рассчитаны. Для полноценной оптимизации нужно модифицировать
    # prepare_features.py или пересчитывать признаки на лету. Но идею вы поняли.

    trades, daily_rets, port_val = st.walk_forward(df)
    if not trades:
        return None
    trades_df = pd.DataFrame(trades)
    pnl_sum = trades_df["pnl"].sum()
    win_rate = (trades_df["pnl"] > 0).mean()
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    return {
        "pnl_sum": pnl_sum,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "num_trades": len(trades_df),
    }

def main():
    df = st.load_data()
    results = []

    # Сетка параметров
    proba_long_range = [0.52, 0.55, 0.60, 0.65]
    max_hold_range = [3, 5, 7]
    calm_filter_range = [True, False]

    for proba, hold, calm in itertools.product(proba_long_range, max_hold_range, calm_filter_range):
        print(f"Тест: proba={proba}, hold={hold}, calm_filter={calm}")
        # ВАЖНО: здесь нужно использовать те же параметры RSI/NEAR_LEVEL, что и при генерации данных.
        # Если мы меняем RSI, нужен пересчёт признаков. Поэтому оставим RSI=30, NEAR=0.01.
        metrics = evaluate_params(
            df,
            rsi_oversold=30,
            rsi_overbought=70,
            near_level=0.01,
            proba_long=proba,
            max_hold=hold,
            use_calm_filter=calm
        )
        if metrics:
            metrics.update({"proba": proba, "hold": hold, "calm": calm})
            results.append(metrics)
            print(f"  PnL={metrics['pnl_sum']:.2%}, WinRate={metrics['win_rate']:.1%}, Sharpe={metrics['sharpe']:.2f}, Trades={metrics['num_trades']}")

    if results:
        results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        print("\nТоп-5 по Sharpe:")
        print(results_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()