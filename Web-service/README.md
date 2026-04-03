# Платформа моделирования СБП

Имитационное моделирование платежной системы с теорией массового обслуживания G/G/c/K и анализом устойчивости на основе ВРПС (Вектор Работоспособности Платежной Системы).

---

## Содержание

- [Архитектура проекта](#архитектура-проекта)
- [Установка](#установка)
- [Основные компоненты](#основные-компоненты)
  - [Симуляция банковских транзакций](#симуляция-банковских-транзакций)
  - [Sender (Отправитель трафика)](#sender-отправитель-трафика)
  - [Receiver (Приёмник трафика)](#receiver-приёмник-трафика)
  - [Dashboard ВРПС](#dashboard-врпс)
- [ВРПС — Вектор Работоспособности Платежной Системы](#врпс--вектор-работоспособности-платежной-системы)
- [Тестирование](#тестирование)
  - [Тесты модулей ВРПС](#тесты-модулей-врпс)
  - [Тесты генераторов трафика](#тесты-генераторов-трафика)
  - [Автономная симуляция СБП](#автономная-симуляция-сбп)
- [Быстрый старт](#быстрый-старт)
- [Зависимости](#зависимости)

---

## Архитектура проекта

```
sbp-platform/
│
├── app/                              # Основной код приложения
│   │
│   ├── analysis/                     # Аналитические модули
│   │   ├── vrps.py                   # ВРПС - расчёт 5 компонент вектора состояния
│   │   ├── lstm_predictor.py         # LSTM прогнозирование состояния системы
│   │   ├── kalman_filter.py          # Фильтр Калмана для сглаживания метрик
│   │   ├── decision_matrix.py        # Матрица принятия решений (9 режимов)
│   │   ├── stability_monitor.py      # Главный монитор устойчивости
│   │   ├── queueing_ggck.py          # G/G/c/K модель очередей
│   │   ├── queuing.py                # Теория массового обслуживания
│   │   ├── statistical.py            # Статистический анализ
│   │   ├── sla.py                    # SLA анализ
│   │   ├── sla_validator.py          # Валидация SLA метрик
│   │   └── correlation.py            # Корреляционный анализ
│   │
│   ├── attacks/                      # Генераторы атак и аномалий
│   │   ├── generator.py              # Генератор трафика (normal/flood/burst)
│   │   ├── patterns.py               # Паттерны атак (SQLi, XSS, SSRF, etc.)
│   │   └── realistic.py              # Реалистичные DDoS сценарии
│   │
│   ├── traffic/                      # Генераторы трафика
│   │   ├── generator.py              # TrafficFlowGenerator
│   │   ├── background.py             # Фоновый трафик N_bg(t)
│   │   └── anomalous.py              # Аномальный трафик N_anom(t)
│   │
│   ├── models/                       # Pydantic модели данных
│   │   ├── traffic_flow.py           # Модели потоков
│   │   └── queueing.py               # Модели очередей
│   │
│   ├── integrations/                 # Внешние интеграции
│   │   └── zabbix.py                 # Zabbix API клиент
│   │
│   ├── storage/                      # Хранение данных
│   │   └── database.py               # PostgreSQL операции
│   │
│   ├── services/                     # Сервисы
│   │   └── metrics.py                # Сбор и агрегация метрик
│   │
│   ├── sender.py                     # FastAPI Sender (порт 5000)
│   ├── receiver.py                   # FastAPI Receiver (порт 5001)
│   ├── dashboard.py                  # Flask Dashboard ВРПС (порт 5050)
│   ├── database.py                   # Подключение к БД
│   ├── models.py                     # SQLAlchemy модели
│   ├── schemas.py                    # Pydantic схемы
│   └── config.py                     # Конфигурация приложения
│
├── tools/                            # Вспомогательные скрипты
│   ├── test_vrps.py                  # Тесты модулей ВРПС
│   ├── test_traffic_flow.py          # Тесты генераторов трафика
│   ├── run_sbp_simulation.py         # Автономный симулятор СБП
│   └── run_simulation.py             # CLI обёртка симуляции
│
├── data/transactions/                # JSON датасеты транзакций
├── reports/                          # Графики и отчёты
│
├── generate_transactions.py          # Генератор банковских транзакций
├── run_bank_simulation.py            # Симулятор банковских транзакций
├── run_sender.py                     # Запуск Sender
├── run_receiver.py                   # Запуск Receiver
├── run_dashboard.py                  # Запуск Dashboard
├── setup_db.py                       # Инициализация БД
└── requirements.txt                  # Зависимости
```

---

## Установка

### 1. Клонирование и настройка окружения

```bash
git clone https://github.com/meloch287/Web-service.git
cd Web-service

# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/Mac)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка базы данных

```bash
# Создайте PostgreSQL базу данных
# Скопируйте .env.example в .env и настройте параметры подключения

# Инициализация таблиц
python setup_db.py
```

### 3. Конфигурация (.env)

```env
DB_USER=vtsk
DB_PASSWORD=1234
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vtsk_db

SENDER_PORT=5000
RECEIVER_PORT=5001
RECEIVER_URL=http://127.0.0.1:5001
```

---

## Основные компоненты

### Симуляция банковских транзакций

Система моделирует реальный поток банковских транзакций с учётом:
- Суточного распределения активности (пики в 10:00, 13:00, 19:00)
- Различных типов аномалий (SQL-инъекции, XSS, фрод)
- Математических распределений (Пуассон, Парето, экспоненциальное)

**Генерация датасета транзакций:**
```bash
python generate_transactions.py --transactions 10000 --anomaly-ratio 0.15
```

Параметры:
- `--users` — количество пользователей (по умолчанию 100)
- `--transactions` — количество транзакций (по умолчанию 30000)
- `--anomaly-ratio` — доля аномальных транзакций (0-1, по умолчанию 0.15)

**Запуск симуляции:**
```bash
python run_bank_simulation.py --input data/transactions/transactions_XXX.json --rps 100
```

---

### Sender (Отправитель трафика)

FastAPI сервис для генерации и отправки тестового трафика на Receiver.

**Запуск:**
```bash
python run_sender.py
# или
uvicorn app.sender:app --host 0.0.0.0 --port 5000
```

**Возможности:**
- Режимы трафика: `normal`, `flood`, `burst`, `slowloris`, `gradual`, `mixed`
- Настраиваемый RPS (requests per second)
- Генерация вредоносных запросов (SQL injection, XSS, path traversal, etc.)
- Сбор метрик: latency, throughput, block rate

**API эндпоинты:**
- `POST /start` — запуск тестовой сессии
- `GET /status/{session_id}` — статус сессии
- `GET /report/{session_id}` — полный отчёт
- `POST /stop/{session_id}` — остановка сессии

---

### Receiver (Приёмник трафика)

FastAPI сервис, имитирующий защищённую платежную систему с WAF.

**Запуск:**
```bash
python run_receiver.py
# или
uvicorn app.receiver:app --host 0.0.0.0 --port 5001
```

**Возможности:**
- Детекция атак по сигнатурам (SQL injection, XSS, path traversal, command injection, XXE, SSRF)
- Блокировка вредоносных User-Agent (sqlmap, nikto, nmap, etc.)
- Запись событий защиты в БД
- Статистика по сессиям

**API эндпоинты:**
- `POST /receive` — приём трафика
- `GET /stats/{session_id}` — статистика сессии
- `GET /events/{session_id}` — события защиты

---

### Dashboard ВРПС

Flask веб-интерфейс для визуализации состояния системы в реальном времени.

**Запуск:**
```bash
python run_dashboard.py
# Открыть http://localhost:5050
```

**Отображает:**
- 5 компонент вектора ВРПС (C, L, Q, R, A)
- Индекс устойчивости Sust(t)
- Текущий режим реагирования (1-9)
- Графики траектории ВРПС
- Статистику режимов

---

## ВРПС — Вектор Работоспособности Платежной Системы

5-компонентный вектор состояния системы S_norm(t) = [C, L, Q, R, A]:

| Компонент | Описание | Формула |
|-----------|----------|---------|
| **C** (Capacity) | Время обработки транзакции | `C = 1 - (T - T_base) / (T_crit - T_base)` |
| **L** (Load) | Коэффициент загрузки | `L = 1 - (ρ / ρ_thresh)²` |
| **Q** (Quality) | Качество обслуживания | `Q = 1 - P_block / P_thresh` |
| **R** (Resources) | Утилизация ресурсов | `R = 1 - U / U_crit` |
| **A** (Anomaly) | Доля нормального трафика | `A = 1 - N_anom / (N_anom + N_bg)` |

**Индекс устойчивости:**
```
Sust(t) = 0.25·C + 0.20·L + 0.35·Q + 0.10·R + 0.10·A
```

**Статусы системы:**
- `HEALTHY` — Sust > 0.8
- `DEGRADED` — 0.5 ≤ Sust ≤ 0.8
- `CRITICAL` — Sust < 0.5

---

## Тестирование

### Тесты модулей ВРПС

Проверка всех аналитических компонентов:

```bash
python tools/test_vrps.py
```

Тестирует:
1. **VRPSCalculator** — расчёт 5 компонент вектора
2. **LSTMPredictor** — прогнозирование состояния
3. **KalmanFilter** — фильтрация и сглаживание
4. **DecisionMatrix** — матрица принятия решений
5. **StabilityMonitor** — интеграционный тест монитора

### Тесты генераторов трафика

Проверка моделей трафика и теории массового обслуживания:

```bash
python tools/test_traffic_flow.py
```

Тестирует:
- Генерацию фонового трафика N_bg(t)
- Суперпозицию N(t) = N_bg(t) + N_anom(t)
- G/G/c/K модель очередей
- SLA валидацию

### Автономная симуляция СБП

Полная симуляция без сети (всё в памяти):

```bash
python tools/run_sbp_simulation.py --servers 10 --queue 1000 --lambda 50 --hours 24
```

Параметры:
- `--servers` — количество серверов (c)
- `--queue` — ёмкость очереди (K)
- `--rate` — скорость обслуживания μ (tx/s)
- `--lambda` — интенсивность входящего потока λ (tx/s)
- `--hours` — длительность симуляции

---

## Быстрый старт

```bash
# 1. Генерация транзакций
python generate_transactions.py --transactions 10000 --anomaly-ratio 0.15

# 2. Запуск receiver (в отдельном терминале)
python run_receiver.py

# 3. Запуск симуляции
python run_bank_simulation.py --rps 100

# 4. Dashboard мониторинга ВРПС (опционально)
python run_dashboard.py
```

---

## Зависимости

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
sqlalchemy>=2.0.25
asyncpg>=0.30.0
httpx>=0.26.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
orjson>=3.9.10
aiofiles>=23.2.1
numpy>=1.24.0
scipy>=1.11.0
python-dotenv>=1.0.0
matplotlib>=3.7.0
aiohttp>=3.9.0
hypothesis>=6.100.0
flask
```
