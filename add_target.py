"""
add_target.py — добавляет будущую доходность и проверяет предсказательную силу bounce_score
Вход:  data/processed/bounce_enriched_*.parquet
Выход: data/processed/bounce_enriched_with_target_*.parquet
"""

import os
import glob
import pandas as pd
import numpy as np
from config import *

DATA_PROCESSED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")

def load_enriched():
    files = sorted(glob.glob(os.path.join(DATA_PROCESSED, "bounce_enriched_*.parquet")))
    if not files:
        raise FileNotFoundError("Нет bounce_enriched_*.parquet в data/processed/")
    df = pd.read_parquet(files[-1])
    df.columns = [str(c).lower() for c in df.columns]
    return df

def add_future_returns(df, lookahead_days=1):
    """Добавляет будущую доходность ret_next_N и направление dir_next_N."""
    df = df.copy()
    if "ticker" not in df.columns:
        raise RuntimeError("Колонка 'ticker' отсутствует — нельзя посчитать доходность по тикерам.")

    # доходность на N дней вперёд
    df[f"ret_next_{lookahead_days}"] = (
        df.groupby("ticker")["close"].shift(-lookahead_days) / df["close"] - 1
    )
    # направление: 1 = рост, 0 = падение
    df[f"dir_next_{lookahead_days}"] = (df[f"ret_next_{lookahead_days}"] > 0).astype(float)
    return df

def quantile_bins_safe(series, n_quantiles=5):
    """
    Строит границы квантилей и метки, корректно обрабатывая дубликаты.
    Возвращает: bins (границы), labels (метки), edges (уникальные границы).
    """
    q = series.quantile(np.linspace(0, 1, n_quantiles + 1)).drop_duplicates()
    # Если уникальных квантилей мало (например, все значения одинаковые), возвращаем дефолт
    if len(q) < 2:
        return [-np.inf, np.inf], ["Q1"], q

    edges = q.tolist()
    # labels должны быть на 1 меньше, чем границ
    labels = [f"Q{i}" for i in range(1, len(edges))]
    return edges, labels, q

def check_score_predictiveness(df, score_col="bounce_score_long", ret_col="ret_next_1"):
    """Считает корреляцию и среднюю доходность по квантилям скора."""
    if score_col not in df.columns or ret_col not in df.columns:
        print(f"⚠️ Нет колонок {score_col} или {ret_col} — пропускаем проверку.")
        return

    mask = df[score_col].notna() & df[ret_col].notna()
    sub = df.loc[mask]
    if len(sub) == 0:
        print(f"⚠️ Нет валидных строк для проверки {score_col}/{ret_col}")
        return

    corr = sub[score_col].corr(sub[ret_col])
    print(f"\nКорреляция {score_col} ↔ {ret_col}: {corr:.4f}")

    # Безопасное построение квантилей
    edges, labels, _ = quantile_bins_safe(sub[score_col], n_quantiles=5)
    if len(labels) == 1:
        # Все значения одинаковые — нельзя разбить на группы
        print("⚠️ Все значения скоринга одинаковы — нельзя построить квантили.")
        return

    sub["score_quantile"] = pd.cut(sub[score_col], bins=edges, labels=labels, include_lowest=True)

    agg = sub.groupby("score_quantile")[ret_col].agg(["count", "mean", "std"])
    agg["mean_annualized"] = agg["mean"] * 252
    print("\nСредняя доходность по квантилям скоринга:")
    print(agg.round(6))

    return corr, agg

def main():
    print("=" * 60)
    print("add_target.py — добавление таргета и проверка скоринга")
    print("=" * 60)

    df = load_enriched()
    print(f"Загружено строк: {len(df)}")
    print(f"Тикеров: {df['ticker'].nunique()}")
    print(f"Колонки: {list(df.columns)}")

    df = add_future_returns(df, lookahead_days=1)
    df = add_future_returns(df, lookahead_days=3)
    print("✅ Добавлены ret_next_1, ret_next_3, dir_next_1, dir_next_3")

    if "bounce_score_long" in df.columns:
        check_score_predictiveness(df, "bounce_score_long", "ret_next_1")
    else:
        print("⚠️ Нет колонки bounce_score_long — пропускаем проверку для long.")

    if "bounce_score_short" in df.columns:
        check_score_predictiveness(df, "bounce_score_short", "ret_next_1")
    else:
        print("⚠️ Нет колонки bounce_score_short — пропускаем проверку для short.")

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_PROCESSED, f"bounce_enriched_with_target_{ts}.parquet")
    df.to_parquet(out_path, index=False)

    print(f"\n✅ Датасет с таргетом сохранён: {out_path}")
    print("Готово!")

if __name__ == "__main__":
    main()
