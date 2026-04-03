#!/usr/bin/env python3
"""
Автономный симулятор Системы Быстрых Платежей (СБП).
Моделирование платежной системы с теорией массового обслуживания G/G/c/K.
Не использует сеть - всё в памяти.

Не используется в основном потоке обмена sender→receiver.
"""
import sys
sys.path.insert(0, '..')

import asyncio
import random
import uuid
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy import stats
import argparse
import matplotlib.pyplot as plt

from app.analysis.queuing import (
    QueuingTheoryAnalyzer, PaymentSystemMarkov, PaymentAnomalyType,
    TransactionState, QueueingMetrics
)

NUM_SERVERS = 10
QUEUE_CAPACITY = 1000
SERVICE_RATE = 100.0
SIMULATION_HOURS = 24


@dataclass
class PaymentTransaction:
    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float
    currency: str = "RUB"
    timestamp: datetime = None
    state: TransactionState = TransactionState.QUEUED
    anomaly_type: int = PaymentAnomalyType.NORMAL
    processing_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    is_completed: bool = False
    is_rejected: bool = False
    rejection_reason: str = ""


class PaymentSystemSimulator:
    def __init__(self, num_servers: int, queue_capacity: int, service_rate: float):
        self.num_servers = num_servers
        self.queue_capacity = queue_capacity
        self.service_rate = service_rate
        self.queuing = QueuingTheoryAnalyzer(num_servers, queue_capacity, service_rate)
        self.markov = PaymentSystemMarkov()
        self.queue: List[PaymentTransaction] = []
        self.processing: List[PaymentTransaction] = []
        self.completed: List[PaymentTransaction] = []
        self.rejected: List[PaymentTransaction] = []
        self.stats = {
            "total_transactions": 0, "completed": 0, "rejected": 0, "timeout": 0,
            "total_queue_time_ms": 0, "total_processing_time_ms": 0,
            "by_hour": {h: {"arrived": 0, "completed": 0, "rejected": 0} for h in range(24)},
            "by_anomaly": {i: {"count": 0, "completed": 0, "rejected": 0} for i in range(8)},
            "p_block_series": [], "n_tot_series": [], "rho_series": [],
            "latencies": [], "queue_lengths": []
        }
        self.session_id = f"SBP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    def generate_normal_transaction(self, timestamp: datetime) -> PaymentTransaction:
        amount = max(100, np.random.lognormal(mean=8, sigma=1.5))
        return PaymentTransaction(
            transaction_id=f"SBP-{uuid.uuid4().hex[:12].upper()}",
            sender_account=f"4081781000{random.randint(10000000, 99999999)}",
            receiver_account=f"4081781000{random.randint(10000000, 99999999)}",
            amount=round(min(amount, 1000000), 2),
            timestamp=timestamp,
            anomaly_type=PaymentAnomalyType.NORMAL
        )
    
    def generate_anomaly_transaction(self, timestamp: datetime, anomaly_type: int) -> PaymentTransaction:
        tx = self.generate_normal_transaction(timestamp)
        tx.anomaly_type = anomaly_type
        if anomaly_type == PaymentAnomalyType.MICRO_TRANSACTION:
            tx.amount = round(random.uniform(1, 100), 2)
        elif anomaly_type == PaymentAnomalyType.HIGH_AMOUNT:
            tx.amount = round(random.uniform(500000, 5000000), 2)
        elif anomaly_type == PaymentAnomalyType.VELOCITY_SPIKE:
            tx.amount = round(random.uniform(100, 1000), 2)
        elif anomaly_type == PaymentAnomalyType.HIGH_VOLATILITY:
            tx.amount = round(random.uniform(1, 1000000), 2)
        return tx
    
    def daily_load_distribution(self, hour: int) -> float:
        primary = stats.norm.pdf(hour, loc=13, scale=2.5)
        secondary = stats.norm.pdf(hour, loc=19, scale=2) * 0.6
        morning = stats.norm.pdf(hour, loc=10, scale=1.5) * 0.4
        return primary + secondary + morning
    
    def generate_transaction_arrivals(self, base_lambda: float, duration_hours: int) -> List[Dict]:
        arrivals = []
        base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        hour_weights = [self.daily_load_distribution(h) for h in range(24)]
        max_weight = max(hour_weights)
        for hour in range(duration_hours):
            hour_lambda = base_lambda * (hour_weights[hour % 24] / max_weight)
            num_arrivals = np.random.poisson(hour_lambda * 3600)
            for _ in range(num_arrivals):
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                microsecond = random.randint(0, 999999)
                ts = base_date + timedelta(hours=hour, minutes=minute, seconds=second, microseconds=microsecond)
                is_anomaly = random.random() < 0.15
                if is_anomaly:
                    anomaly_type = random.choice([
                        PaymentAnomalyType.MICRO_TRANSACTION,
                        PaymentAnomalyType.HIGH_AMOUNT,
                        PaymentAnomalyType.VELOCITY_SPIKE,
                        PaymentAnomalyType.HIGH_VOLATILITY
                    ])
                    tx = self.generate_anomaly_transaction(ts, anomaly_type)
                else:
                    tx = self.generate_normal_transaction(ts)
                arrivals.append({"timestamp": ts, "transaction": tx})
        arrivals.sort(key=lambda x: x["timestamp"])
        return arrivals

    def process_transaction(self, tx: PaymentTransaction, current_load: float) -> PaymentTransaction:
        metrics = self.queuing.analyze_system(current_load)
        self.stats["rho_series"].append(metrics.rho)
        self.stats["p_block_series"].append(metrics.p_block)
        self.stats["queue_lengths"].append(len(self.queue))
        
        if len(self.queue) >= self.queue_capacity:
            tx.state = TransactionState.REJECTED
            tx.is_rejected = True
            tx.rejection_reason = "queue_full"
            self.markov.record_transition(TransactionState.QUEUED, TransactionState.REJECTED)
            return tx
        
        if random.random() < metrics.p_block:
            tx.state = TransactionState.REJECTED
            tx.is_rejected = True
            tx.rejection_reason = "system_overload"
            self.markov.record_transition(TransactionState.QUEUED, TransactionState.REJECTED)
            return tx
        
        queue_time = metrics.e_wait * 1000 * random.uniform(0.5, 1.5)
        tx.queue_time_ms = queue_time
        self.markov.record_transition(TransactionState.QUEUED, TransactionState.PROCESSING)
        
        processing_time = (1 / self.service_rate) * 1000 * random.uniform(0.8, 1.2)
        tx.processing_time_ms = processing_time
        
        if tx.anomaly_type == PaymentAnomalyType.HIGH_AMOUNT:
            if random.random() < 0.3:
                tx.state = TransactionState.REJECTED
                tx.is_rejected = True
                tx.rejection_reason = "amount_limit_exceeded"
                self.markov.record_transition(TransactionState.PROCESSING, TransactionState.REJECTED)
                return tx
        
        if tx.anomaly_type == PaymentAnomalyType.VELOCITY_SPIKE:
            if random.random() < 0.2:
                tx.state = TransactionState.REJECTED
                tx.is_rejected = True
                tx.rejection_reason = "rate_limit"
                self.markov.record_transition(TransactionState.PROCESSING, TransactionState.REJECTED)
                return tx
        
        if queue_time + processing_time > 5000:
            tx.state = TransactionState.TIMEOUT
            tx.is_rejected = True
            tx.rejection_reason = "timeout"
            self.markov.record_transition(TransactionState.PROCESSING, TransactionState.TIMEOUT)
            return tx
        
        tx.state = TransactionState.COMPLETED
        tx.is_completed = True
        self.markov.record_transition(TransactionState.PROCESSING, TransactionState.COMPLETED)
        return tx
    
    def run_simulation(self, base_lambda: float = 50.0, duration_hours: int = 24):
        print(f"\n{'='*70}")
        print(f"СИМУЛЯЦИЯ СИСТЕМЫ БЫСТРЫХ ПЛАТЕЖЕЙ (СБП)")
        print(f"{'='*70}")
        print(f"Сессия: {self.session_id}")
        print(f"Серверов: {self.num_servers}")
        print(f"Ёмкость очереди: {self.queue_capacity}")
        print(f"Скорость обслуживания: {self.service_rate} tx/s")
        print(f"Базовая интенсивность: {base_lambda} tx/s")
        print(f"{'='*70}\n")
        
        print("Генерация транзакций...")
        arrivals = self.generate_transaction_arrivals(base_lambda, duration_hours)
        print(f"Сгенерировано {len(arrivals)} транзакций")
        
        print("\nОбработка транзакций...")
        window_size = 100
        current_window = []
        
        for i, arrival in enumerate(arrivals):
            tx = arrival["transaction"]
            hour = arrival["timestamp"].hour
            self.stats["total_transactions"] += 1
            self.stats["by_hour"][hour]["arrived"] += 1
            self.stats["by_anomaly"][tx.anomaly_type]["count"] += 1
            self.stats["n_tot_series"].append(1)
            
            current_window.append(arrival["timestamp"])
            if len(current_window) > window_size:
                current_window.pop(0)
            
            if len(current_window) > 1:
                time_span = (current_window[-1] - current_window[0]).total_seconds()
                current_load = len(current_window) / max(time_span, 0.001)
            else:
                current_load = base_lambda
            
            tx = self.process_transaction(tx, current_load)
            
            if tx.is_completed:
                self.stats["completed"] += 1
                self.stats["by_hour"][hour]["completed"] += 1
                self.stats["by_anomaly"][tx.anomaly_type]["completed"] += 1
                self.completed.append(tx)
                total_latency = tx.queue_time_ms + tx.processing_time_ms
                self.stats["latencies"].append(total_latency)
                self.stats["total_queue_time_ms"] += tx.queue_time_ms
                self.stats["total_processing_time_ms"] += tx.processing_time_ms
            else:
                self.stats["rejected"] += 1
                self.stats["by_hour"][hour]["rejected"] += 1
                self.stats["by_anomaly"][tx.anomaly_type]["rejected"] += 1
                self.rejected.append(tx)
                if tx.state == TransactionState.TIMEOUT:
                    self.stats["timeout"] += 1
            
            if (i + 1) % 10000 == 0:
                print(f"  Обработано {i+1}/{len(arrivals)} транзакций...")
        
        self.print_results()
        self.generate_charts()

    def print_results(self):
        print(f"\n{'='*70}")
        print(f"РЕЗУЛЬТАТЫ СИМУЛЯЦИИ СБП")
        print(f"{'='*70}")
        
        total = self.stats["total_transactions"]
        completed = self.stats["completed"]
        rejected = self.stats["rejected"]
        
        print(f"\n--- Общая статистика ---")
        print(f"Всего транзакций: {total}")
        print(f"Успешно обработано: {completed} ({completed/total*100:.2f}%)")
        print(f"Отклонено: {rejected} ({rejected/total*100:.2f}%)")
        print(f"Таймауты: {self.stats['timeout']}")
        
        d_loss = self.queuing.calculate_d_loss(
            self.stats["p_block_series"], 
            self.stats["n_tot_series"]
        )
        
        print(f"\n--- Метрики теории массового обслуживания ---")
        print(f"D_loss (доля потерянных): {d_loss:.4f} ({d_loss*100:.2f}%)")
        
        if self.stats["rho_series"]:
            avg_rho = np.mean(self.stats["rho_series"])
            max_rho = max(self.stats["rho_series"])
            print(f"Средний коэффициент загрузки ρ: {avg_rho:.4f}")
            print(f"Максимальный ρ: {max_rho:.4f}")
        
        if self.stats["p_block_series"]:
            avg_p_block = np.mean(self.stats["p_block_series"])
            print(f"Средняя P_block: {avg_p_block:.4f}")
        
        if self.stats["latencies"]:
            lats = self.stats["latencies"]
            print(f"\n--- Время обработки (мс) ---")
            print(f"Среднее: {np.mean(lats):.2f}")
            print(f"P50: {np.percentile(lats, 50):.2f}")
            print(f"P95: {np.percentile(lats, 95):.2f}")
            print(f"P99: {np.percentile(lats, 99):.2f}")
            avg_queue = self.stats["total_queue_time_ms"] / max(completed, 1)
            avg_proc = self.stats["total_processing_time_ms"] / max(completed, 1)
            print(f"Среднее время в очереди: {avg_queue:.2f} мс")
            print(f"Среднее время обработки: {avg_proc:.2f} мс")
        
        print(f"\n--- По типам аномалий ---")
        for anomaly_type in range(8):
            data = self.stats["by_anomaly"][anomaly_type]
            if data["count"] > 0:
                success_rate = data["completed"] / data["count"] * 100
                name = PaymentAnomalyType.to_russian(anomaly_type)
                print(f"{name}: {data['count']} транзакций, успех: {success_rate:.1f}%")
        
        stationary = self.markov.get_stationary_distribution()
        print(f"\n--- Марковская модель (стационарное распределение) ---")
        for state, prob in stationary.items():
            print(f"  {state.value}: {prob:.4f}")
        
        completion_prob = self.markov.get_completion_probability(steps=10)
        print(f"Вероятность успешного завершения (10 шагов): {completion_prob:.4f}")
        
        print(f"\n--- Почасовое распределение ---")
        max_arrived = max(h["arrived"] for h in self.stats["by_hour"].values())
        for hour in range(24):
            data = self.stats["by_hour"][hour]
            bar_len = int(data["arrived"] / max(max_arrived, 1) * 30)
            success_rate = data["completed"] / max(data["arrived"], 1) * 100
            print(f"{hour:02d}:00 | {'█' * bar_len} {data['arrived']} (успех: {success_rate:.0f}%)")
        
        print(f"{'='*70}")

    def generate_charts(self):
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Результаты симуляции СБП (G/G/c/K)\nСессия: {self.session_id}', fontsize=14, fontweight='bold')
        
        ax1 = fig.add_subplot(2, 3, 1)
        hours = list(range(24))
        completed = [self.stats["by_hour"][h]["completed"] for h in hours]
        rejected = [self.stats["by_hour"][h]["rejected"] for h in hours]
        ax1.bar(hours, completed, color='#2ecc71', label='Успешные')
        ax1.bar(hours, rejected, bottom=completed, color='#e74c3c', label='Отклонённые')
        ax1.set_xlabel('Час')
        ax1.set_ylabel('Количество транзакций')
        ax1.set_title('Распределение транзакций за 24 часа')
        ax1.legend()
        ax1.set_xticks(range(0, 24, 2))
        
        ax2 = fig.add_subplot(2, 3, 2)
        if self.stats["rho_series"]:
            x = np.linspace(0, 24, len(self.stats["rho_series"]))
            ax2.plot(x, self.stats["rho_series"], 'b-', alpha=0.7, label='ρ(t)')
            ax2.axhline(y=1.0, color='r', linestyle='--', label='ρ = 1 (перегрузка)')
            ax2.axhline(y=0.8, color='orange', linestyle='--', label='ρ = 0.8 (предел)')
            ax2.set_xlabel('Время (часы)')
            ax2.set_ylabel('Коэффициент загрузки ρ')
            ax2.set_title('Коэффициент загрузки системы ρ(t)')
            ax2.legend(fontsize=8)
            ax2.set_ylim(0, max(1.5, max(self.stats["rho_series"]) * 1.1))
        
        ax3 = fig.add_subplot(2, 3, 3)
        if self.stats["p_block_series"]:
            x = np.linspace(0, 24, len(self.stats["p_block_series"]))
            ax3.plot(x, self.stats["p_block_series"], 'r-', alpha=0.7)
            ax3.fill_between(x, self.stats["p_block_series"], alpha=0.3, color='red')
            ax3.set_xlabel('Время (часы)')
            ax3.set_ylabel('P_block')
            ax3.set_title('Вероятность блокировки P_block(t)')
            d_loss = self.queuing.calculate_d_loss(self.stats["p_block_series"], self.stats["n_tot_series"])
            ax3.text(0.05, 0.95, f'D_loss = {d_loss:.4f}', transform=ax3.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        ax4 = fig.add_subplot(2, 3, 4)
        if self.stats["latencies"]:
            ax4.hist(self.stats["latencies"], bins=50, color='#3498db', edgecolor='white', alpha=0.7)
            ax4.axvline(np.mean(self.stats["latencies"]), color='red', linestyle='--', linewidth=2, label=f'E[W] = {np.mean(self.stats["latencies"]):.1f}мс')
            ax4.axvline(np.percentile(self.stats["latencies"], 95), color='orange', linestyle='--', linewidth=2, label=f'P95 = {np.percentile(self.stats["latencies"], 95):.1f}мс')
            ax4.set_xlabel('Время обработки (мс)')
            ax4.set_ylabel('Частота')
            ax4.set_title('Распределение времени обработки')
            ax4.legend(fontsize=8)
        
        ax5 = fig.add_subplot(2, 3, 5)
        anomaly_names = [PaymentAnomalyType.to_russian(i) for i in range(8)]
        counts = [self.stats["by_anomaly"][i]["count"] for i in range(8)]
        success_rates = [self.stats["by_anomaly"][i]["completed"] / max(self.stats["by_anomaly"][i]["count"], 1) * 100 for i in range(8)]
        colors = ['#2ecc71' if r >= 90 else '#f39c12' if r >= 70 else '#e74c3c' for r in success_rates]
        bars = ax5.barh(anomaly_names, success_rates, color=colors)
        ax5.set_xlabel('Успешность (%)')
        ax5.set_title('Успешность по типам транзакций')
        ax5.set_xlim(0, 110)
        for bar, rate, count in zip(bars, success_rates, counts):
            ax5.annotate(f'{rate:.0f}% (n={count})', xy=(rate + 2, bar.get_y() + bar.get_height()/2), va='center', fontsize=8)
        
        ax6 = fig.add_subplot(2, 3, 6)
        if self.stats["queue_lengths"]:
            x = np.linspace(0, 24, len(self.stats["queue_lengths"]))
            ax6.plot(x, self.stats["queue_lengths"], 'purple', alpha=0.7)
            ax6.axhline(y=self.queue_capacity, color='red', linestyle='--', label=f'K = {self.queue_capacity}')
            ax6.set_xlabel('Время (часы)')
            ax6.set_ylabel('Длина очереди')
            ax6.set_title('Длина очереди E[L_q](t)')
            ax6.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_dir = "../reports"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/sbp_{self.session_id}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Графики сохранены: {filename}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="СБП Simulator")
    parser.add_argument("--servers", type=int, default=10, help="Количество серверов (c)")
    parser.add_argument("--queue", type=int, default=1000, help="Ёмкость очереди (K)")
    parser.add_argument("--rate", type=float, default=100.0, help="Скорость обслуживания μ (tx/s)")
    parser.add_argument("--lambda", dest="lambda_rate", type=float, default=50.0, help="Интенсивность λ (tx/s)")
    parser.add_argument("--hours", type=int, default=24, help="Длительность симуляции (часы)")
    args = parser.parse_args()
    
    simulator = PaymentSystemSimulator(args.servers, args.queue, args.rate)
    simulator.run_simulation(args.lambda_rate, args.hours)


if __name__ == "__main__":
    main()
