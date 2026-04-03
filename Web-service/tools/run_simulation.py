#!/usr/bin/env python3
"""
CLI обёртка для запуска тестов через app.main.
Не используется в основном потоке обмена.
"""
import sys
sys.path.insert(0, '..')

import asyncio
from app.main import run_test

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    requests = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    malicious = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    attacks = sys.argv[5] if len(sys.argv) > 5 else "sql_injection,xss"
    
    asyncio.run(run_test(
        mode=mode,
        total_requests=requests,
        requests_per_second=rps,
        malicious_ratio=malicious,
        attack_types=attacks
    ))
