"""
config.py — v7: только шорты, RR=1.5, скоринг убран из фичей
"""

import os

# === Пути ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_RESULTS = os.path.join(PROJECT_ROOT, "data", "results")

for d in [DATA_RAW, DATA_PROCESSED, DATA_RESULTS]:
    os.makedirs(d, exist_ok=True)

# === Тикеры MOEX ===
TICKERS = [
    "GAZP", "LKOH", "NVTK", "ROSN", "TATN", "GMKN", "NLMK", "ALRS",
    "PLZL", "CHMF", "SBER", "VTBR", "MGNT", "YDEX",
]

# === Данные ===
FETCH_DAYS = 730

# === Индикаторы ===
ATR_PERIOD = 14
MA_FAST = 50
MA_SLOW = 200
RSI_PERIOD = 14
VOL_LOOKBACK = 20
BB_PERIOD = 20
BB_STD = 2

# === Уровни ===
LEVEL_TOL = 0.01
SWING_LOOKBACK = 20
USE_BB_LEVELS = False
USE_SWING_LEVELS = True
USE_MA_LEVELS = False

# === Bounce-скоринг ===
SCORE_WEIGHTS = {
    "score_close":        0.30,
    "score_volume":       0.15,
    "score_confirm":      0.10,
    "score_penetration":  0.10,
    "score_trend":        0.05,
    "score_tests":        0.20,
    "score_rsi":          0.10,
}

# === Стоп / Тейк / Трейлинг ===
ATR_MULT_STOP = 2.0
RR_RATIO = 1.5
TRAIL_AFTER_RR = 0.7
TRAIL_ATR_MULT = 1.5
TX_COST = 0.001
MAX_HOLDING_DAYS = 10

# === Направления ===
DIRECTIONS = ["short"]

# === Трендовый фильтр ===
USE_TREND_FILTER = True

# === Walk-Forward ===
WF_N_SPLITS = 5
CONFIDENCE_THRESHOLD = 0.55

# === XGBoost ===
XGB_PARAMS = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "random_state": 42,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# === Признаки ===
# Скоринг убран — он обратно коррелирует с PnL (Spearman = -0.108, p=0.042)
FEATURES_WITH_SCORE = [
    "rsi_14", "vol_ratio", "atr_14",
    "ma_50", "ma_200", "close_to_ma50",
]

FEATURES_WITHOUT_SCORE = ["rsi_14", "vol_ratio", "atr_14", "ma_50", "ma_200", "close_to_ma50"]

MIN_TRADES = 30
