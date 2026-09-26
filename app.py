"""
app.py — Веб-приложение для управления bounce-стратегией
"""

import os
import subprocess
import sys
import glob
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import joblib

from config import *

app = Flask(__name__)

# Убедимся, что нужные папки существуют
for d in [DATA_RESULTS, GRAPHS_DIR, MODELS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


def run_script(script_name):
    """Запускает python-скрипт и возвращает результат"""
    log_file = os.path.join(LOG_DIR, f"{script_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, script_name],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
    return result.returncode, log_file


@app.route("/")
def index():
    """Главная страница — дашборд"""
    # Получаем список последних результатов
    trades_files = sorted(glob.glob(os.path.join(DATA_RESULTS, "trades_*.csv")), reverse=True)
    equity_files = sorted(glob.glob(os.path.join(DATA_RESULTS, "equity_*.csv")), reverse=True)
    latest_trades = trades_files[0] if trades_files else None
    latest_equity = equity_files[0] if equity_files else None

    # Загружаем метрики из последнего equity
    metrics = {}
    if latest_equity:
        eq = pd.read_csv(latest_equity)
        daily_rets = eq["daily_return"]
        total_return = (1 + daily_rets).prod() - 1
        sharpe = daily_rets.mean() / daily_rets.std() * (252**0.5) if daily_rets.std() > 0 else 0
        metrics = {
            "total_return": f"{total_return:.2%}",
            "sharpe": f"{sharpe:.2f}",
            "num_days": len(eq),
        }

    return render_template("index.html",
                           metrics=metrics,
                           latest_trades=os.path.basename(latest_trades) if latest_trades else None,
                           latest_equity=os.path.basename(latest_equity) if latest_equity else None)


@app.route("/run", methods=["POST"])
def run_pipeline():
    """Запуск полного цикла: fetch → prepare → strategy"""
    steps = ["fetch_data.py", "prepare_features.py", "strategy_bounce_v2.py"]
    logs = []
    for step in steps:
        code, log_file = run_script(step)
        logs.append({"script": step, "returncode": code, "log": os.path.basename(log_file)})
        if code != 0:
            return jsonify({"status": "error", "steps": logs})
    return jsonify({"status": "success", "steps": logs})


@app.route("/run_single", methods=["POST"])
def run_single():
    """Запуск одного скрипта"""
    script = request.json.get("script")
    if script not in ["fetch_data.py", "prepare_features.py", "strategy_bounce_v2.py"]:
        return jsonify({"status": "error", "message": "Неизвестный скрипт"})
    code, log_file = run_script(script)
    return jsonify({"status": "success" if code == 0 else "error", "log": os.path.basename(log_file)})


@app.route("/results")
def results():
    """Страница с последними сделками и графиками"""
    trades_files = sorted(glob.glob(os.path.join(DATA_RESULTS, "trades_*.csv")), reverse=True)
    if trades_files:
        trades_df = pd.read_csv(trades_files[0])
        trades = trades_df.to_dict(orient="records")
    else:
        trades = []

    graphs = sorted(glob.glob(os.path.join(GRAPHS_DIR, "*.png")), reverse=True)
    latest_graphs = [os.path.basename(g) for g in graphs[:4]]

    return render_template("results.html", trades=trades, graphs=latest_graphs)


@app.route("/graphs/<filename>")
def graph(filename):
    return send_from_directory(GRAPHS_DIR, filename)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Страница настройки параметров (редактирование config.py)"""
    if request.method == "POST":
        # Простое обновление ключевых параметров через форму
        # В реальности нужно аккуратно записывать в config.py
        # Здесь для примера покажем только чтение
        pass

    # Читаем текущие значения из config.py
    with open("config.py", "r", encoding="utf-8") as f:
        config_text = f.read()

    return render_template("settings.html", config_text=config_text)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)