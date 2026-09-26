"""
run_daily.py — автоматический запуск всех этапов стратегии:
1) fetch_data.py
2) prepare_features.py
3) strategy_bounce_v2.py

Логи сохраняются в папку logs/
"""

import os
import subprocess
import sys
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

def run_script(script_name):
    """Запускает python-скрипт и сохраняет его вывод в лог-файл"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        print(f"--- Запуск {script_name} ---")
        print(f"Лог: {log_file}")
        result = subprocess.run(
            [sys.executable, script_name],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            print(f"Ошибка при выполнении {script_name} (код {result.returncode})")
            return False
    print(f"{script_name} завершён успешно.\n")
    return True

def main():
    print("=== Ежедневный прогон стратегии ===\n")
    steps = ["fetch_data.py", "prepare_features.py", "strategy_bounce_v2.py"]
    for step in steps:
        success = run_script(step)
        if not success:
            print(f"Произошла ошибка на шаге {step}. Прогон остановлен.")
            sys.exit(1)
    print("Все шаги выполнены успешно!")

if __name__ == "__main__":
    main()