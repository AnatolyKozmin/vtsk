#!/usr/bin/env python3
"""
Запуск ВРПС Dashboard
"""

import argparse
import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.dashboard import run_dashboard


def main():
    parser = argparse.ArgumentParser(description='ВРПС Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5050, help='Port (default: 5050)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     ВРПС Dashboard - Мониторинг устойчивости СБП          ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Компоненты:                                              ║
    ║  • C (Capacity)  - время обработки транзакции             ║
    ║  • L (Load)      - коэффициент загрузки                   ║
    ║  • Q (Quality)   - вероятность блокировки                 ║
    ║  • R (Resources) - утилизация ресурсов                    ║
    ║  • A (Anomaly)   - доля аномального трафика               ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    run_dashboard(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
