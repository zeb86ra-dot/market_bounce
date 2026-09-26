import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_LIVE = os.path.join(PROJECT_ROOT, "data", "live")
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_RESULTS = os.path.join(PROJECT_ROOT, "data", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "results_graphs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

TICKERS = [
    "GAZP", "LKOH", "NVTK", "ROSN", "TATN", "GMKN", "NLMK", "ALRS",
    "PLZL", "CHMF", "SBER", "VTBR", "MGNT", "YDEX",
    "AFKS", "IRAO", "SNGS", "SMLT", "RUAL", "POLY", "FIVE", "LSRG",
    "AFLT", "PIKK", "MOEX", "CBOM",
    "MAGN", "RASP", "NMTP", "UNAC", "KMAZ", "MSNG", "OGKB", "TGKA",
]

HORIZON = 5
MAX_HOLD_DAYS = 5

MA_PERIOD = 20
MA_LONG = 50
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
VOL_PERCENTILE = 75

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
NEAR_LEVEL_THRESHOLD = 0.01
STOP_LOSS = 0.05
TAKE_PROFIT = 0.08

TRAIN_WINDOW = 120
TEST_WINDOW = 21
STEP = 21

PROBA_LONG = 0.65
PROBA_SHORT = 0.40
MAX_POSITIONS = 7
USE_REL_TARGET = False
COMMISSION = 0.001

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
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

FEATURE_COLS = [
    "ma_ratio", "ma_trend", "price_above_ma",
    "rsi", "rsi_oversold", "rsi_overbought",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pctb", "dist_to_lower_bb", "dist_to_upper_bb", "bb_squeeze",
    "ret_1d", "ret_5d", "ret_10d", "momentum",
    "drawdown", "max_dd_20d",
    "vol_ratio", "vol_decline", "vol_trend",
    "vol_20d", "vol_skew", "high_low_range",
    "near_support", "near_resistance", "pullback_depth",
    "bounce_long_signal", "bounce_short_signal",
    "dist_to_5d_low", "dist_to_5d_high",
    "breadth", "sector_rank",
    "pullback_uptrend", "close_position", "gap", "close_to_ma20", "rsi_ma",
    "body_ratio",
]

FETCH_DAYS = 730
