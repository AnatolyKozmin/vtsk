# Tools

Вспомогательные скрипты, не используемые в основном потоке обмена sender→receiver.

## Содержимое

| Файл | Назначение |
|------|------------|
| `test_traffic_flow.py` | Unit-тесты для TrafficFlowGenerator и GGcKQueueingSystem |
| `run_sbp_simulation.py` | Автономный симулятор СБП (теория очередей G/G/c/K) |
| `run_simulation.py` | CLI обёртка для app.main.run_test() |

## Запуск

```bash
cd tools

# Тесты модулей
python test_traffic_flow.py

# Автономная симуляция СБП (без сети)
python run_sbp_simulation.py --servers 10 --queue 1000 --lambda 50

# CLI тест через app.main
python run_simulation.py normal 1000 100 0.1 "sql_injection,xss"
```

## Основной поток

Для основного потока используйте скрипты в корне проекта:

```bash
# 1. Генерация транзакций
python generate_transactions.py --transactions 10000

# 2. Запуск receiver
python run_receiver.py

# 3. Запуск симуляции
python run_bank_simulation.py
```
