#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# Справочные данные
# ============================================================

BICS = [
    "044525225",
    "044525593",
    "044525000",
    "044525999",
    "040349700",
    "042202603",
]

# Типы транзакций:
# C2C  - клиент -> клиент
# C2B  - клиент -> бизнес
# B2C  - бизнес -> клиент
# ME2ME - между своими счетами
TRN_TYPES = ["C2C", "C2B", "B2C", "ME2ME"]

COMMON_NARRATIVES_BY_TYPE = {
    "C2C": [
        "Перевод другу",
        "Возврат долга",
        "Личный перевод",
        "Перевод по СБП",
    ],
    "C2B": [
        "Оплата услуг",
        "Оплата заказа",
        "Покупка товара",
        "Оплата подписки",
    ],
    "B2C": [
        "Возврат средств",
        "Выплата клиенту",
        "Компенсация",
        "Зачисление",
    ],
    "ME2ME": [
        "Перевод между счетами",
        "Перевод себе",
        "Пополнение счета",
        "Внутренний перевод",
    ],
}

MALE_FIRST_NAMES = [
    "Иван", "Петр", "Алексей", "Дмитрий", "Сергей", "Андрей",
    "Николай", "Михаил", "Владимир", "Евгений",
]
FEMALE_FIRST_NAMES = [
    "Мария", "Елена", "Ольга", "Анна", "Наталья", "Светлана",
    "Ирина", "Екатерина", "Татьяна", "Юлия",
]

MALE_LAST_NAMES = [
    "Иванов", "Петров", "Сидоров", "Козлов", "Новиков",
    "Морозов", "Волков", "Соколов", "Лебедев", "Кузнецов",
]
FEMALE_LAST_NAMES = [
    "Иванова", "Петрова", "Сидорова", "Козлова", "Новикова",
    "Морозова", "Волкова", "Соколова", "Лебедева", "Кузнецова",
]

MALE_PATRONYMICS = [
    "Иванович", "Петрович", "Сергеевич", "Дмитриевич", "Алексеевич",
    "Николаевич", "Михайлович", "Владимирович", "Евгеньевич",
]
FEMALE_PATRONYMICS = [
    "Ивановна", "Петровна", "Сергеевна", "Дмитриевна", "Алексеевна",
    "Николаевна", "Михайловна", "Владимировна", "Евгеньевна",
]

CITIES = [
    "Москва", "Санкт-Петербург", "Казань", "Новосибирск",
    "Екатеринбург", "Нижний Новгород", "Самара", "Краснодар",
]

BUSINESS_NAMES = [
    "ООО Ромашка",
    "ООО Вектор",
    "ООО АльфаМаркет",
    "АО СервисПлюс",
    "ООО БыстраяОплата",
    "ООО ГородСервис",
    "ООО ТехноТорг",
    "ООО КомфортПэй",
]


# ============================================================
# Конфигурация
# ============================================================

@dataclass
class GeneratorConfig:
    output_path: str
    start_date: str = "2026-04-01"
    days: int = 7

    normal_count: int = 560_000
    normal_peak_count: int = 35_000
    exponential_count: int = 35_000
    poisson_count: int = 35_000
    pareto_count: int = 35_000

    total_clients: int = 12_000
    total_beneficiaries: int = 5_000

    hot_pool_peak: int = 200
    hot_pool_exponential: int = 40
    hot_pool_poisson: int = 80
    hot_pool_pareto: int = 60

    velocity_threshold: int = 10
    velocity_window_seconds: int = 300

    seed: int = 42


# ============================================================
# Доменные сущности
# ============================================================

@dataclass
class BeneficiaryProfile:
    beneficiary_id: int
    entity_kind: str  # "person" | "business"
    pam: str
    full_name: str
    bic: str


@dataclass
class ClientProfile:
    client_id: str
    gender: str
    pam: str
    full_name: str
    account: str
    address: str
    payer_bic: str

    active_start_hour: int
    active_end_hour: int
    typical_amount_mean: float
    typical_amount_std: float

    own_beneficiary_id: int
    favorite_beneficiary_ids: List[int] = field(default_factory=list)

    tx_count: int = 0
    amount_sum: float = 0.0
    last_tx_time: datetime | None = None

    def mean_amount_so_far(self) -> float:
        if self.tx_count == 0:
            return self.typical_amount_mean
        return self.amount_sum / self.tx_count


# ============================================================
# Базовые утилиты
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)


def make_account() -> str:
    return "40817" + "".join(str(random.randint(0, 9)) for _ in range(15))


def make_client_id(i: int) -> str:
    return f"{i:08d}"


def make_trn_id(seq: int) -> str:
    return f"{seq:08d}"


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def day_background_weight(hour: float) -> float:
    return (
        0.40 * gaussian(hour, 10.0, 1.5)
        + 1.00 * gaussian(hour, 13.0, 2.5)
        + 0.60 * gaussian(hour, 19.0, 2.0)
        + 0.02
    )


def weighted_choice(items: List[int], weights: List[float]) -> int:
    return random.choices(items, weights=weights, k=1)[0]


def split_total_by_days(total: int, days: int) -> List[int]:
    base = total // days
    rem = total % days
    out = [base] * days
    for i in range(rem):
        out[i] += 1
    return out


def build_address() -> str:
    city = random.choice(CITIES)
    house = random.randint(1, 200)
    flat = random.randint(1, 300)
    return f"г. {city}, ул. Ленина, д. {house}, кв. {flat}"


# ============================================================
# Корректная генерация ФИО по полу
# ============================================================

def build_person_identity(gender: str) -> Tuple[str, str]:
    if gender == "male":
        first_name = random.choice(MALE_FIRST_NAMES)
        last_name = random.choice(MALE_LAST_NAMES)
        patronymic = random.choice(MALE_PATRONYMICS)
    else:
        first_name = random.choice(FEMALE_FIRST_NAMES)
        last_name = random.choice(FEMALE_LAST_NAMES)
        patronymic = random.choice(FEMALE_PATRONYMICS)

    full_name = f"{last_name} {first_name} {patronymic}"
    pam = f"{last_name} {first_name[0]}."
    return pam, full_name


def build_business_identity() -> Tuple[str, str]:
    name = random.choice(BUSINESS_NAMES)
    return name, name


# ============================================================
# Создание профилей получателей и клиентов
# ============================================================

def create_beneficiaries(total_beneficiaries: int) -> Dict[int, BeneficiaryProfile]:
    beneficiaries: Dict[int, BeneficiaryProfile] = {}

    for i in range(total_beneficiaries):
        entity_kind = "business" if random.random() < 0.35 else "person"

        if entity_kind == "person":
            gender = random.choice(["male", "female"])
            pam, full_name = build_person_identity(gender)
        else:
            pam, full_name = build_business_identity()

        beneficiaries[i] = BeneficiaryProfile(
            beneficiary_id=i,
            entity_kind=entity_kind,
            pam=pam,
            full_name=full_name,
            bic=random.choice(BICS),
        )

    return beneficiaries


def create_clients(config: GeneratorConfig, beneficiaries: Dict[int, BeneficiaryProfile]) -> Dict[str, ClientProfile]:
    clients: Dict[str, ClientProfile] = {}
    all_beneficiary_ids = list(beneficiaries.keys())

    for i in range(config.total_clients):
        client_id = make_client_id(i)
        gender = random.choice(["male", "female"])
        pam, full_name = build_person_identity(gender)

        active_start = random.randint(6, 11)
        active_end = random.randint(18, 23)

        typical_amount_mean = random.uniform(2_000, 60_000)
        typical_amount_std = random.uniform(500, 10_000)

        own_beneficiary_id = random.choice(all_beneficiary_ids)

        favorite_count = random.randint(3, 15)
        favorite_ids = random.sample(all_beneficiary_ids, k=favorite_count)

        clients[client_id] = ClientProfile(
            client_id=client_id,
            gender=gender,
            pam=pam,
            full_name=full_name,
            account=make_account(),  # ВАЖНО: закреплено за клиентом один раз
            address=build_address(),
            payer_bic=random.choice(BICS),
            active_start_hour=active_start,
            active_end_hour=active_end,
            typical_amount_mean=typical_amount_mean,
            typical_amount_std=typical_amount_std,
            own_beneficiary_id=own_beneficiary_id,
            favorite_beneficiary_ids=favorite_ids,
        )

    return clients


def build_active_clients_index(clients: Dict[str, ClientProfile]) -> Dict[int, List[str]]:
    by_hour: Dict[int, List[str]] = {h: [] for h in range(24)}
    all_ids = list(clients.keys())

    for client in clients.values():
        for h in range(client.active_start_hour, client.active_end_hour + 1):
            if 0 <= h < 24:
                by_hour[h].append(client.client_id)

    for h in range(24):
        if not by_hour[h]:
            by_hour[h] = all_ids.copy()

    return by_hour


# ============================================================
# Генерация времени событий
# ============================================================

def generate_background_timestamps(base_date: datetime, count: int) -> List[datetime]:
    hours = list(range(24))
    weights = [day_background_weight(h) for h in hours]

    out: List[datetime] = []
    for _ in range(count):
        hour = weighted_choice(hours, weights)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        out.append(base_date.replace(hour=hour, minute=minute, second=second, microsecond=0))
    return out


def pick_random_window_start_hour() -> float:
    return round(random.uniform(8.0, 21.0), 2)


def generate_normal_peak_timestamps(base_date: datetime, count: int, tau_hour: float, gamma_hours: float) -> List[datetime]:
    out: List[datetime] = []
    while len(out) < count:
        hour = random.gauss(tau_hour, gamma_hours / 2)
        if 0 <= hour < 24:
            h = int(hour)
            m = int((hour - h) * 60)
            s = random.randint(0, 59)
            out.append(base_date.replace(hour=h, minute=m, second=s, microsecond=0))
    return out


def generate_exponential_timestamps(base_date: datetime, count: int, start_hour: float, duration_minutes: int) -> List[datetime]:
    out: List[datetime] = []
    window_sec = duration_minutes * 60

    start_ts = base_date.replace(
        hour=int(start_hour),
        minute=int((start_hour % 1) * 60),
        second=0,
        microsecond=0,
    )

    for _ in range(count):
        offset = min(int(random.expovariate(1 / max(1, window_sec / 4))), window_sec - 1)
        out.append(start_ts + timedelta(seconds=offset))
    return out


def generate_poisson_timestamps(base_date: datetime, count: int, start_hour: float, duration_minutes: int, bursts: int) -> List[datetime]:
    start_ts = base_date.replace(
        hour=int(start_hour),
        minute=int((start_hour % 1) * 60),
        second=0,
        microsecond=0,
    )
    end_ts = start_ts + timedelta(minutes=duration_minutes)
    total_sec = int((end_ts - start_ts).total_seconds())

    burst_centers = sorted(random.sample(range(total_sec), k=min(bursts, max(1, total_sec // 60))))

    out: List[datetime] = []
    for _ in range(count):
        center = random.choice(burst_centers)
        offset = int(random.gauss(center, 40))
        offset = max(0, min(offset, total_sec - 1))
        out.append(start_ts + timedelta(seconds=offset))
    return out


def generate_pareto_timestamps(base_date: datetime, count: int, start_hour: float, duration_minutes: int) -> List[datetime]:
    start_ts = base_date.replace(
        hour=int(start_hour),
        minute=int((start_hour % 1) * 60),
        second=0,
        microsecond=0,
    )
    window_sec = duration_minutes * 60

    out: List[datetime] = []
    for _ in range(count):
        x = random.paretovariate(1.5)
        offset = int(min(window_sec - 1, (x - 1.0) * (window_sec / 5)))
        out.append(start_ts + timedelta(seconds=offset))
    return out


# ============================================================
# Логика типов транзакций
# ============================================================

def choose_trn_type(label: int) -> str:
    """
    Тип транзакции теперь не фиксирован.
    Он зависит от сценария, но без жёсткой "подсказки" модели.
    """
    if label == 2:
        # exponential — чаще мелкие/частые клиентские платежи
        return random.choices(
            ["C2C", "C2B", "ME2ME"],
            weights=[0.35, 0.45, 0.20],
            k=1,
        )[0]

    if label == 3:
        # poisson — пачки, часто C2C или C2B
        return random.choices(
            ["C2C", "C2B", "ME2ME"],
            weights=[0.45, 0.40, 0.15],
            k=1,
        )[0]

    if label == 4:
        # pareto — крупные суммы, чаще C2C/C2B/B2C
        return random.choices(
            ["C2C", "C2B", "B2C"],
            weights=[0.40, 0.35, 0.25],
            k=1,
        )[0]

    # normal / normal peak
    return random.choices(
        ["C2C", "C2B", "B2C", "ME2ME"],
        weights=[0.45, 0.30, 0.10, 0.15],
        k=1,
    )[0]


def choose_narrative(trn_type: str) -> str:
    return random.choice(COMMON_NARRATIVES_BY_TYPE[trn_type])


# ============================================================
# Содержимое транзакций по классам
# ============================================================

def hot_probability(label: int) -> float:
    if label == 0:
        return 0.02
    if label == 1:
        return 0.10
    if label == 2:
        return 0.90
    if label == 3:
        return 0.60
    if label == 4:
        return 0.75
    return 0.10


def sample_normal_like_amount(client: ClientProfile, trn_type: str) -> int:
    base_multiplier = {
        "C2C": 1.00,
        "C2B": 0.90,
        "B2C": 1.10,
        "ME2ME": 1.20,
    }[trn_type]

    amount = random.gauss(
        client.typical_amount_mean * base_multiplier,
        client.typical_amount_std,
    )
    return int(min(max(amount, 10), 300_000))


def sample_exponential_amount(trn_type: str) -> int:
    base = {
        "C2C": 180.0,
        "C2B": 250.0,
        "ME2ME": 400.0,
        "B2C": 600.0,
    }[trn_type]

    amount = random.expovariate(1 / base)
    return int(min(max(amount, 10), 3_000))


def sample_poisson_amount(client: ClientProfile, trn_type: str) -> int:
    multiplier = {
        "C2C": 1.00,
        "C2B": 0.95,
        "B2C": 1.15,
        "ME2ME": 1.10,
    }[trn_type]

    amount = random.gauss(
        client.typical_amount_mean * multiplier,
        client.typical_amount_std * 1.2,
    )
    return int(min(max(amount, 10), 500_000))


def sample_pareto_amount(client: ClientProfile, trn_type: str) -> int:
    heavy_multiplier = {
        "C2C": 1.0,
        "C2B": 1.2,
        "B2C": 1.4,
        "ME2ME": 1.1,
    }[trn_type]

    if random.random() < 0.20:
        x = random.paretovariate(1.6)
        return int(min(5_000_000, max(100_000, client.typical_amount_mean * heavy_multiplier * (3 + x * 3))))

    amount = random.gauss(
        client.typical_amount_mean * 1.3 * heavy_multiplier,
        client.typical_amount_std * 1.8,
    )
    return int(min(max(amount, 10), 1_500_000))


def choose_amount_for_event(label: int, client: ClientProfile, trn_type: str) -> int:
    if label == 0:
        return sample_normal_like_amount(client, trn_type)
    if label == 1:
        return sample_normal_like_amount(client, trn_type)
    if label == 2:
        return sample_exponential_amount(trn_type)
    if label == 3:
        return sample_poisson_amount(client, trn_type)
    if label == 4:
        return sample_pareto_amount(client, trn_type)
    return sample_normal_like_amount(client, trn_type)


# ============================================================
# Выбор клиента и получателя
# ============================================================

def choose_client_for_event(
    label: int,
    event_hour: int,
    active_clients_by_hour: Dict[int, List[str]],
    hot_pools: Dict[int, List[str]],
) -> str:
    active_candidates = active_clients_by_hour[event_hour]

    if random.random() > hot_probability(label):
        return random.choice(active_candidates)

    hot_candidates = hot_pools.get(label, [])
    if not hot_candidates:
        return random.choice(active_candidates)

    active_set = set(active_candidates)
    hot_active = [cid for cid in hot_candidates if cid in active_set]
    if hot_active:
        return random.choice(hot_active)

    return random.choice(hot_candidates)


def choose_beneficiary_for_event(
    client: ClientProfile,
    beneficiaries: Dict[int, BeneficiaryProfile],
    trn_type: str,
    label: int,
    day_poisson_beneficiary_id: int,
) -> BeneficiaryProfile:
    # Poisson: усиливаем доминирующего получателя
    if label == 3 and random.random() < 0.70:
        return beneficiaries[day_poisson_beneficiary_id]

    # Между своими счетами — чаще собственный счёт
    if trn_type == "ME2ME":
        return beneficiaries[client.own_beneficiary_id]

    # C2B — логичнее чаще платить бизнесу
    if trn_type == "C2B":
        business_beneficiaries = [b for b in beneficiaries.values() if b.entity_kind == "business"]
        if business_beneficiaries and random.random() < 0.80:
            return random.choice(business_beneficiaries)

    # B2C — логичнее чаще физлицу
    if trn_type == "B2C":
        person_beneficiaries = [b for b in beneficiaries.values() if b.entity_kind == "person"]
        if person_beneficiaries and random.random() < 0.80:
            return random.choice(person_beneficiaries)

    # Обычный сценарий: любимые получатели
    if random.random() < 0.80:
        bid = random.choice(client.favorite_beneficiary_ids)
        return beneficiaries[bid]

    return beneficiaries[random.choice(list(beneficiaries.keys()))]


# ============================================================
# Онлайн-флаги
# ============================================================

class OnlineFlags:
    def __init__(self, velocity_threshold: int, velocity_window_seconds: int):
        self.velocity_threshold = velocity_threshold
        self.velocity_window_seconds = velocity_window_seconds
        self.per_client_times: Dict[str, deque] = defaultdict(deque)
        self.seen_pairs: set[Tuple[str, int]] = set()

    def update_and_get_flags(self, client_id: str, beneficiary_id: int, ts: datetime) -> Dict[str, bool]:
        q = self.per_client_times[client_id]
        q.append(ts)

        cutoff = ts - timedelta(seconds=self.velocity_window_seconds)
        while q and q[0] < cutoff:
            q.popleft()

        velocity_anomaly = len(q) > self.velocity_threshold

        pair = (client_id, beneficiary_id)
        is_new_pair = pair not in self.seen_pairs
        self.seen_pairs.add(pair)

        return {
            "velocity_anomaly": velocity_anomaly,
            "is_new_pair_meta": is_new_pair,
        }


# ============================================================
# Построение JSON
# ============================================================

def build_payload(
    ts: datetime,
    seq_id: int,
    trn_type: str,
    client: ClientProfile,
    beneficiary: BeneficiaryProfile,
    amount: int,
    label: int,
    flags: Dict[str, bool],
) -> Dict:
    return {
        "Data": {
            "CurrentTimestamp": iso_z(ts),
            "TrnId": make_trn_id(seq_id),
            "TrnType": trn_type,
            "PayerData": {
                "ClientId": client.client_id,
                "PAM": client.pam,
                "FullName": client.full_name,
                "Account": client.account,  # закреплён за клиентом
                "Address": client.address,
                "Direction": "Out",
                "PayerBIC": client.payer_bic,
            },
            "BeneficiaryData": {
                "PAM": beneficiary.pam,
                "FullName": beneficiary.full_name,
                "BeneficiaryBIC": beneficiary.bic,
            },
            "Amount": str(amount),
            "Currency": "RUB",
            "Narrative": choose_narrative(trn_type),
        },
        "Meta": {
            "base_label": label,
            "is_anomaly": label != 0,
            "velocity_anomaly": flags["velocity_anomaly"],
            "is_new_pair_meta": flags["is_new_pair_meta"],
        },
    }


# ============================================================
# События одного дня
# ============================================================

def build_day_events(
    day_date: datetime,
    normal_count: int,
    normal_peak_count: int,
    exponential_count: int,
    poisson_count: int,
    pareto_count: int,
) -> List[Tuple[datetime, int]]:
    events: List[Tuple[datetime, int]] = []

    bg = generate_background_timestamps(day_date, normal_count)
    events.extend((ts, 0) for ts in bg)

    peak_tau = random.choice([10.0, 13.0, 19.0]) + random.uniform(-1.0, 1.0)
    peak_gamma = random.uniform(0.8, 1.8)

    exp_start = pick_random_window_start_hour()
    exp_duration = random.randint(20, 60)

    pois_start = pick_random_window_start_hour()
    pois_duration = random.randint(30, 90)
    pois_bursts = random.randint(6, 18)

    pareto_start = pick_random_window_start_hour()
    pareto_duration = random.randint(40, 120)

    events.extend((ts, 1) for ts in generate_normal_peak_timestamps(day_date, normal_peak_count, peak_tau, peak_gamma))
    events.extend((ts, 2) for ts in generate_exponential_timestamps(day_date, exponential_count, exp_start, exp_duration))
    events.extend((ts, 3) for ts in generate_poisson_timestamps(day_date, poisson_count, pois_start, pois_duration, pois_bursts))
    events.extend((ts, 4) for ts in generate_pareto_timestamps(day_date, pareto_count, pareto_start, pareto_duration))

    events.sort(key=lambda x: x[0])
    return events


# ============================================================
# Основной генератор
# ============================================================

class FinalSBPGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        set_seed(config.seed)

        self.beneficiaries = create_beneficiaries(config.total_beneficiaries)
        self.clients = create_clients(config, self.beneficiaries)
        self.active_clients_by_hour = build_active_clients_index(self.clients)

        client_ids = list(self.clients.keys())
        self.hot_pools = {
            1: random.sample(client_ids, k=min(config.hot_pool_peak, len(client_ids))),
            2: random.sample(client_ids, k=min(config.hot_pool_exponential, len(client_ids))),
            3: random.sample(client_ids, k=min(config.hot_pool_poisson, len(client_ids))),
            4: random.sample(client_ids, k=min(config.hot_pool_pareto, len(client_ids))),
        }

        self.flags_engine = OnlineFlags(
            velocity_threshold=config.velocity_threshold,
            velocity_window_seconds=config.velocity_window_seconds,
        )

        self.seq_id = 1
        self.summary_counts = defaultdict(int)
        self.summary_velocity = 0

    def generate(self) -> None:
        out_path = Path(self.config.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        start_date = datetime.fromisoformat(self.config.start_date).replace(
            tzinfo=timezone.utc,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        normal_by_day = split_total_by_days(self.config.normal_count, self.config.days)
        peak_by_day = split_total_by_days(self.config.normal_peak_count, self.config.days)
        exp_by_day = split_total_by_days(self.config.exponential_count, self.config.days)
        pois_by_day = split_total_by_days(self.config.poisson_count, self.config.days)
        pareto_by_day = split_total_by_days(self.config.pareto_count, self.config.days)

        with out_path.open("w", encoding="utf-8") as f:
            for day_idx in range(self.config.days):
                day_date = start_date + timedelta(days=day_idx)

                day_events = build_day_events(
                    day_date=day_date,
                    normal_count=normal_by_day[day_idx],
                    normal_peak_count=peak_by_day[day_idx],
                    exponential_count=exp_by_day[day_idx],
                    poisson_count=pois_by_day[day_idx],
                    pareto_count=pareto_by_day[day_idx],
                )

                day_poisson_beneficiary_id = random.choice(list(self.beneficiaries.keys()))

                for ts, label in day_events:
                    client_id = choose_client_for_event(
                        label=label,
                        event_hour=ts.hour,
                        active_clients_by_hour=self.active_clients_by_hour,
                        hot_pools=self.hot_pools,
                    )
                    client = self.clients[client_id]

                    trn_type = choose_trn_type(label)

                    beneficiary = choose_beneficiary_for_event(
                        client=client,
                        beneficiaries=self.beneficiaries,
                        trn_type=trn_type,
                        label=label,
                        day_poisson_beneficiary_id=day_poisson_beneficiary_id,
                    )

                    amount = choose_amount_for_event(label, client, trn_type)

                    flags = self.flags_engine.update_and_get_flags(
                        client_id=client.client_id,
                        beneficiary_id=beneficiary.beneficiary_id,
                        ts=ts,
                    )

                    payload = build_payload(
                        ts=ts,
                        seq_id=self.seq_id,
                        trn_type=trn_type,
                        client=client,
                        beneficiary=beneficiary,
                        amount=amount,
                        label=label,
                        flags=flags,
                    )

                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

                    client.tx_count += 1
                    client.amount_sum += amount
                    client.last_tx_time = ts

                    self.summary_counts[label] += 1
                    if flags["velocity_anomaly"]:
                        self.summary_velocity += 1

                    self.seq_id += 1

                print(
                    f"[day {day_idx + 1}/{self.config.days}] "
                    f"written {len(day_events)} rows, total so far = {self.seq_id - 1}"
                )

        print("\n=== GENERATION SUMMARY ===")
        print("Output file:", out_path)
        print("Rows written:", self.seq_id - 1)
        print("Counts by label:", dict(sorted(self.summary_counts.items())))
        print("Velocity flagged:", self.summary_velocity)


# ============================================================
# CLI
# ============================================================

def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Final SBP transactions generator")

    parser.add_argument("--out", type=str, default="data/sbp_final.jsonl")
    parser.add_argument("--start-date", type=str, default="2026-04-01")
    parser.add_argument("--days", type=int, default=7)

    parser.add_argument("--normal", type=int, default=560_000)
    parser.add_argument("--normal-peak", type=int, default=35_000)
    parser.add_argument("--exponential", type=int, default=35_000)
    parser.add_argument("--poisson", type=int, default=35_000)
    parser.add_argument("--pareto", type=int, default=35_000)

    parser.add_argument("--clients", type=int, default=12_000)
    parser.add_argument("--beneficiaries", type=int, default=5_000)

    parser.add_argument("--hot-pool-peak", type=int, default=200)
    parser.add_argument("--hot-pool-exponential", type=int, default=40)
    parser.add_argument("--hot-pool-poisson", type=int, default=80)
    parser.add_argument("--hot-pool-pareto", type=int, default=60)

    parser.add_argument("--velocity-threshold", type=int, default=10)
    parser.add_argument("--velocity-window-seconds", type=int, default=300)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return GeneratorConfig(
        output_path=args.out,
        start_date=args.start_date,
        days=args.days,

        normal_count=args.normal,
        normal_peak_count=args.normal_peak,
        exponential_count=args.exponential,
        poisson_count=args.poisson,
        pareto_count=args.pareto,

        total_clients=args.clients,
        total_beneficiaries=args.beneficiaries,

        hot_pool_peak=args.hot_pool_peak,
        hot_pool_exponential=args.hot_pool_exponential,
        hot_pool_poisson=args.hot_pool_poisson,
        hot_pool_pareto=args.hot_pool_pareto,

        velocity_threshold=args.velocity_threshold,
        velocity_window_seconds=args.velocity_window_seconds,

        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    generator = FinalSBPGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()