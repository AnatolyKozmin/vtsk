#!/usr/bin/env python3
"""
Симулятор банковских транзакций.
Читает предгенерированные транзакции из JSON и отправляет на receiver.

Использование:
  1. Сначала сгенерируй транзакции: python generate_transactions.py
  2. Затем запусти симуляцию: python run_bank_simulation.py --input data/transactions/transactions_XXX.json
"""
import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import List, Dict
import numpy as np
from scipy import stats
import argparse
import matplotlib.pyplot as plt

TARGET_URL = "http://127.0.0.1:5001/receive"

RUSSIAN_LABELS = {
    "sql_injection": "SQL-инъекция",
    "xss": "XSS-атака",
    "fraud_velocity": "Частотная\nаномалия",
    "fraud_amount_anomaly": "Аномалия\nсуммы",
    "fraud_geo_anomaly": "Гео-\nаномалия",
    "normal": "Норма",
}


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
    
    @classmethod
    def from_string(cls, attack_type: str) -> int:
        mapping = {
            "normal": cls.NORMAL,
            "sql_injection": cls.SQL_INJECTION,
            "xss": cls.XSS,
            "fraud_velocity": cls.FRAUD_VELOCITY,
            "fraud_amount_anomaly": cls.FRAUD_AMOUNT,
            "fraud_geo_anomaly": cls.FRAUD_GEO,
            "fraud_time_anomaly": cls.FRAUD_TIME,
            "ddos": cls.DDOS,
            "brute_force": cls.BRUTE_FORCE,
        }
        return mapping.get(attack_type, cls.NORMAL)


class DistributionType:
    POISSON = "poisson"
    PARETO = "pareto"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    
    @classmethod
    def all(cls) -> List[str]:
        return [cls.POISSON, cls.PARETO, cls.EXPONENTIAL]
    
    @classmethod
    def to_russian(cls, dist_type: str) -> str:
        return {
            "poisson": "Пуассон",
            "pareto": "Парето",
            "exponential": "Экспоненц.",
            "normal": "Норм."
        }.get(dist_type, dist_type)


class TransactionDistribution:
    @staticmethod
    def daily_activity_distribution(hour: int) -> float:
        primary = stats.norm.pdf(hour, loc=13, scale=2.5)
        secondary = stats.norm.pdf(hour, loc=19, scale=2) * 0.6
        morning = stats.norm.pdf(hour, loc=10, scale=1.5) * 0.4
        return primary + secondary + morning


class BankSimulator:
    def __init__(self, target_url: str, dataset_path: str = None):
        self.target_url = target_url
        self.dataset_path = dataset_path
        self.dataset = None
        self.transactions = []
        self.session_id = f"BANK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        self.stats = {
            "total_sent": 0,
            "total_received": 0,
            "total_blocked": 0,
            "malicious_sent": 0,
            "malicious_blocked": 0,
            "by_attack_type": {},
            "by_hour": {h: 0 for h in range(24)},
            "by_hour_normal": {h: 0 for h in range(24)},
            "by_hour_anomaly": {h: 0 for h in range(24)},
            "by_hour_poisson": {h: 0 for h in range(24)},
            "by_hour_pareto": {h: 0 for h in range(24)},
            "by_hour_exponential": {h: 0 for h in range(24)},
            "by_distribution": {d: 0 for d in DistributionType.all() + [DistributionType.NORMAL]},
            "latencies": []
        }
    
    def load_dataset(self, filepath: str) -> bool:
        """Загружает датасет из JSON файла"""
        if not os.path.exists(filepath):
            print(f"❌ Файл не найден: {filepath}")
            return False
        
        print(f"Загрузка датасета: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        self.transactions = self.dataset.get("transactions", [])
        meta = self.dataset.get("metadata", {})
        
        print(f"  Session ID: {meta.get('session_id', 'unknown')}")
        print(f"  Транзакций: {len(self.transactions)}")
        print(f"  Пользователей: {meta.get('num_users', 0)}")
        print(f"  Доля аномалий: {meta.get('anomaly_ratio', 0)*100:.1f}%")
        
        return True
    
    async def send_transaction(self, session: aiohttp.ClientSession, transaction: Dict) -> Dict:
        """Отправляет одну транзакцию на receiver"""
        request_data = {
            "request_id": transaction["transaction_id"],
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "is_malicious": transaction.get("is_malicious", False),
            "attack_type": transaction.get("attack_type", "normal"),
            "anomaly_code": transaction.get("anomaly_code", AnomalyType.NORMAL),
            "payload": transaction,
            "headers": {
                "User-Agent": "BankApp/2.0",
                "X-Device-ID": transaction.get("device_fingerprint", "unknown")
            }
        }
        
        try:
            start = datetime.utcnow()
            async with session.post(self.target_url, json=request_data, timeout=10) as resp:
                elapsed = (datetime.utcnow() - start).total_seconds() * 1000
                result = await resp.json()
                
                # Парсим час из timestamp транзакции
                try:
                    tx_hour = datetime.fromisoformat(transaction["timestamp"]).hour
                except:
                    tx_hour = 0
                
                return {
                    "success": True,
                    "transaction_id": transaction["transaction_id"],
                    "attack_type": transaction.get("attack_type", "normal"),
                    "is_malicious": transaction.get("is_malicious", False),
                    "was_blocked": result.get("results", [{}])[0].get("was_blocked", False),
                    "latency_ms": elapsed,
                    "hour": tx_hour,
                    "distribution": transaction.get("distribution", DistributionType.NORMAL)
                }
        except Exception as e:
            return {
                "success": False,
                "transaction_id": transaction["transaction_id"],
                "attack_type": transaction.get("attack_type", "normal"),
                "is_malicious": transaction.get("is_malicious", False),
                "was_blocked": False,
                "latency_ms": 0,
                "error": str(e)
            }
    
    async def run_simulation(self, rps: int = 100, batch_size: int = 50):
        """Запускает симуляцию отправки транзакций"""
        if not self.transactions:
            print("❌ Нет транзакций для отправки. Загрузите датасет.")
            return
        
        print(f"\n{'='*70}")
        print(f"BANK TRANSACTION SIMULATION")
        print(f"{'='*70}")
        print(f"Session: {self.session_id}")
        print(f"Target: {self.target_url}")
        print(f"Transactions: {len(self.transactions)}")
        print(f"RPS: {rps}")
        print(f"{'='*70}\n")
        
        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            start_time = datetime.utcnow()
            
            for i in range(0, len(self.transactions), batch_size):
                batch = self.transactions[i:i+batch_size]
                tasks = [self.send_transaction(session, tx) for tx in batch]
                results = await asyncio.gather(*tasks)
                
                for result in results:
                    self.stats["total_sent"] += 1
                    
                    if result["success"]:
                        self.stats["total_received"] += 1
                        self.stats["latencies"].append(result["latency_ms"])
                        self.stats["by_hour"][result["hour"]] += 1
                        
                        dist = result.get("distribution", DistributionType.NORMAL)
                        self.stats["by_distribution"][dist] += 1
                        
                        if result["is_malicious"]:
                            self.stats["by_hour_anomaly"][result["hour"]] += 1
                            if dist == DistributionType.POISSON:
                                self.stats["by_hour_poisson"][result["hour"]] += 1
                            elif dist == DistributionType.PARETO:
                                self.stats["by_hour_pareto"][result["hour"]] += 1
                            elif dist == DistributionType.EXPONENTIAL:
                                self.stats["by_hour_exponential"][result["hour"]] += 1
                        else:
                            self.stats["by_hour_normal"][result["hour"]] += 1
                        
                        if result["was_blocked"]:
                            self.stats["total_blocked"] += 1
                        
                        if result["is_malicious"]:
                            self.stats["malicious_sent"] += 1
                            if result["was_blocked"]:
                                self.stats["malicious_blocked"] += 1
                        
                        attack_type = result["attack_type"]
                        if attack_type not in self.stats["by_attack_type"]:
                            self.stats["by_attack_type"][attack_type] = {"sent": 0, "blocked": 0}
                        self.stats["by_attack_type"][attack_type]["sent"] += 1
                        if result["was_blocked"]:
                            self.stats["by_attack_type"][attack_type]["blocked"] += 1
                
                progress = (i + len(batch)) / len(self.transactions) * 100
                print(f"\rProgress: {progress:.1f}% ({i + len(batch)}/{len(self.transactions)})", end="", flush=True)
                
                await asyncio.sleep(batch_size / rps)
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
        
        self.print_results(elapsed)
    
    def print_results(self, elapsed: float):
        """Выводит результаты симуляции"""
        print(f"\n\n{'='*70}")
        print(f"SIMULATION RESULTS")
        print(f"{'='*70}")
        print(f"Duration: {elapsed:.2f}s")
        print(f"Throughput: {self.stats['total_sent']/elapsed:.1f} tx/s")
        
        print(f"\n--- Transaction Stats ---")
        print(f"Total sent: {self.stats['total_sent']}")
        print(f"Total received: {self.stats['total_received']}")
        print(f"Total blocked: {self.stats['total_blocked']}")
        print(f"Block rate: {self.stats['total_blocked']/max(1,self.stats['total_received'])*100:.1f}%")
        
        if self.stats["latencies"]:
            lats = sorted(self.stats["latencies"])
            print(f"\n--- Latency (ms) ---")
            print(f"Avg: {np.mean(lats):.2f} | Min: {min(lats):.2f} | Max: {max(lats):.2f}")
            print(f"P50: {np.percentile(lats, 50):.2f} | P95: {np.percentile(lats, 95):.2f} | P99: {np.percentile(lats, 99):.2f}")
        
        print(f"\n--- Attack Detection ---")
        print(f"Malicious sent: {self.stats['malicious_sent']}")
        print(f"Malicious blocked: {self.stats['malicious_blocked']}")
        detection_rate = self.stats['malicious_blocked']/max(1,self.stats['malicious_sent'])*100
        print(f"Detection rate: {detection_rate:.1f}%")
        
        normal_sent = self.stats["by_attack_type"].get("normal", {}).get("sent", 0)
        normal_blocked = self.stats["by_attack_type"].get("normal", {}).get("blocked", 0)
        fp_rate = normal_blocked/max(1,normal_sent)*100
        print(f"False positive rate: {fp_rate:.1f}%")
        
        print(f"\n--- By Attack Type ---")
        for attack_type, data in self.stats["by_attack_type"].items():
            rate = data["blocked"]/max(1,data["sent"])*100
            print(f"{attack_type}: sent={data['sent']}, blocked={data['blocked']} ({rate:.1f}%)")
        
        print(f"\n--- Hourly Distribution ---")
        max_count = max(self.stats["by_hour"].values()) if self.stats["by_hour"] else 1
        for hour in range(24):
            count = self.stats["by_hour"][hour]
            bar_len = int(count / max_count * 40)
            print(f"{hour:02d}:00 | {'█' * bar_len} {count}")
        
        print(f"{'='*70}")
        
        if detection_rate < 80:
            print(f"⚠️  WARNING: Detection rate below 80%")
        if fp_rate > 5:
            print(f"⚠️  WARNING: False positive rate above 5%")
        
        self.generate_charts(elapsed, detection_rate, fp_rate)
    
    def generate_charts(self, elapsed: float, detection_rate: float, fp_rate: float):
        """Генерирует графики результатов"""
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Результаты симуляции банковских транзакций\nСессия: {self.session_id}', 
                     fontsize=14, fontweight='bold')
        
        # График 1: Распределение по часам
        ax1 = fig.add_subplot(2, 3, 1)
        hours = list(range(24))
        normal_counts = [self.stats["by_hour_normal"][h] for h in hours]
        poisson_counts = [self.stats["by_hour_poisson"][h] for h in hours]
        pareto_counts = [self.stats["by_hour_pareto"][h] for h in hours]
        exp_counts = [self.stats["by_hour_exponential"][h] for h in hours]
        
        bottom1 = normal_counts
        bottom2 = [n + p for n, p in zip(normal_counts, poisson_counts)]
        bottom3 = [b + pa for b, pa in zip(bottom2, pareto_counts)]
        
        ax1.bar(hours, normal_counts, color='#2ecc71', edgecolor='white', linewidth=0.5, label='Легитимные (норм.)')
        ax1.bar(hours, poisson_counts, bottom=normal_counts, color='#e74c3c', edgecolor='white', linewidth=0.5, label='Аномалии (Пуассон)')
        ax1.bar(hours, pareto_counts, bottom=bottom2, color='#9b59b6', edgecolor='white', linewidth=0.5, label='Аномалии (Парето)')
        ax1.bar(hours, exp_counts, bottom=bottom3, color='#f39c12', edgecolor='white', linewidth=0.5, label='Аномалии (Экспон.)')
        
        # Теоретическое нормальное распределение (одна плавная кривая)
        # Пик в 12:00 (полдень), стд. отклонение 4 часа
        total_transactions = sum(self.stats["by_hour"].values())
        if total_transactions > 0:
            x_smooth = np.linspace(0, 23, 200)
            # Простое нормальное распределение с пиком в середине дня
            theoretical = stats.norm.pdf(x_smooth, loc=12, scale=4)
            # Масштабируем под фактическое количество транзакций
            theoretical_scaled = theoretical / theoretical.max() * max(self.stats["by_hour"].values())
            ax1_twin = ax1.twinx()
            ax1_twin.plot(x_smooth, theoretical_scaled, 'r--', linewidth=2.5, label='Теоретическое')
            ax1_twin.set_ylabel('Теоретическое распределение', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            ax1_twin.legend(loc='upper right', fontsize=6)
        
        ax1.set_xlabel('Время суток')
        ax1.set_ylabel('Количество транзакций')
        ax1.set_title('Распределение транзакций за 24 часа\n(Нормальное распределение)')
        ax1.set_xticks(range(0, 24, 2))
        ax1.legend(loc='upper left', fontsize=6)
        
        # График 2: По типам атак
        ax2 = fig.add_subplot(2, 3, 2)
        attack_types = list(self.stats["by_attack_type"].keys())
        sent_counts = [self.stats["by_attack_type"][t]["sent"] for t in attack_types]
        blocked_counts = [self.stats["by_attack_type"][t]["blocked"] for t in attack_types]
        x_pos = np.arange(len(attack_types))
        width = 0.35
        ax2.bar(x_pos - width/2, sent_counts, width, label='Отправлено', color='#3498db')
        ax2.bar(x_pos + width/2, blocked_counts, width, label='Заблокировано', color='#e74c3c')
        ax2.set_xlabel('Тип аномалии')
        ax2.set_ylabel('Количество')
        ax2.set_title('Детекция по типам аномалий')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([RUSSIAN_LABELS.get(t, t) for t in attack_types], rotation=0, fontsize=8)
        ax2.legend()
        
        # График 3: Ключевые метрики
        ax3 = fig.add_subplot(2, 3, 3)
        metrics = ['Детекция\n(%)', 'Ложные\nсрабатывания', 'Блокировка\n(%)', 'Пропускная\nспособность']
        values = [
            detection_rate,
            fp_rate,
            self.stats['total_blocked']/max(1,self.stats['total_received'])*100,
            self.stats['total_sent']/elapsed
        ]
        colors = [
            '#2ecc71' if detection_rate >= 80 else '#e74c3c',
            '#2ecc71' if fp_rate <= 5 else '#e74c3c',
            '#3498db',
            '#9b59b6'
        ]
        bars = ax3.bar(metrics, values, color=colors, edgecolor='white', linewidth=2)
        ax3.set_ylabel('Значение')
        ax3.set_title('Ключевые метрики эффективности')
        ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5)
        ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, values):
            ax3.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontweight='bold')
        
        # График 4: Латентность
        ax4 = fig.add_subplot(2, 3, 4)
        if self.stats["latencies"]:
            lats = self.stats["latencies"]
            # Гистограмма фактических значений (нормализованная для сравнения с PDF)
            n, bins, patches = ax4.hist(lats, bins=50, color='#3498db', edgecolor='white', 
                                        alpha=0.7, density=True, label='Фактическое')
            
            # Теоретическое нормальное распределение
            mean_lat = np.mean(lats)
            std_lat = np.std(lats)
            x_theory = np.linspace(min(lats), max(lats), 200)
            y_theory = stats.norm.pdf(x_theory, loc=mean_lat, scale=std_lat)
            ax4.plot(x_theory, y_theory, 'r-', linewidth=2.5, 
                    label=f'Теор. норм. (μ={mean_lat:.1f}, σ={std_lat:.1f})')
            
            ax4.axvline(mean_lat, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax4.axvline(np.percentile(lats, 95), color='orange', linestyle='--', linewidth=1.5,
                       label=f'P95: {np.percentile(lats, 95):.1f}мс')
            ax4.set_xlabel('Задержка (мс)')
            ax4.set_ylabel('Плотность вероятности')
            ax4.set_title('Распределение времени отклика\n(факт vs теория)')
            ax4.legend(fontsize=7, loc='upper right')
        
        # График 5: Теоретические распределения
        ax5 = fig.add_subplot(2, 3, 5)
        x_range = np.linspace(0, 10, 200)
        y_norm = stats.norm.pdf(x_range, loc=5, scale=1.5)
        ax5.plot(x_range, y_norm, 'g-', linewidth=2, label='Нормальное')
        ax5.fill_between(x_range, y_norm, alpha=0.3, color='green')
        y_exp = stats.expon.pdf(x_range, scale=2)
        ax5.plot(x_range, y_exp, 'r-', linewidth=2, label='Экспоненциальное')
        ax5.fill_between(x_range, y_exp, alpha=0.3, color='red')
        y_pareto = stats.pareto.pdf(x_range, b=2.0)
        ax5.plot(x_range, y_pareto, 'b-', linewidth=2, label='Парето')
        ax5.fill_between(x_range, y_pareto, alpha=0.3, color='blue')
        ax5.set_xlabel('Значение')
        ax5.set_ylabel('Плотность вероятности')
        ax5.set_title('Математические модели распределений')
        ax5.legend(fontsize=8)
        ax5.set_xlim(0, 10)
        ax5.set_ylim(0, 1)
        
        # График 6: Детекция по типам
        ax6 = fig.add_subplot(2, 3, 6)
        attack_map = {
            'sql_injection': 'SQL-инъекция',
            'xss': 'XSS-атака',
            'fraud_velocity': 'Частотная аномалия',
            'fraud_amount_anomaly': 'Аномалия суммы',
            'fraud_geo_anomaly': 'Гео-аномалия'
        }
        detection_rates = []
        labels_list = []
        for key, label in attack_map.items():
            data = self.stats["by_attack_type"].get(key, {"sent": 0, "blocked": 0})
            rate = data["blocked"] / max(1, data["sent"]) * 100
            detection_rates.append(rate)
            labels_list.append(f"{label}\n(код: {AnomalyType.from_string(key)})")
        
        colors = ['#2ecc71' if r >= 80 else '#e74c3c' for r in detection_rates]
        bars = ax6.barh(labels_list, detection_rates, color=colors, edgecolor='white')
        ax6.set_xlabel('Уровень детекции (%)')
        ax6.set_title('Детекция по типам аномалий')
        ax6.axvline(x=80, color='green', linestyle='--', alpha=0.7)
        ax6.set_xlim(0, 110)
        for bar, rate in zip(bars, detection_rates):
            ax6.annotate(f'{rate:.0f}%', xy=(rate + 2, bar.get_y() + bar.get_height()/2),
                        va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/simulation_{self.session_id}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Графики сохранены: {filename}")
        plt.show()


def find_latest_dataset(directory: str = "data/transactions") -> str:
    """Находит последний сгенерированный датасет"""
    if not os.path.exists(directory):
        return None
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        return None
    
    files.sort(reverse=True)
    return os.path.join(directory, files[0])


async def main():
    parser = argparse.ArgumentParser(description="Bank Transaction Simulator")
    parser.add_argument("--input", type=str, default=None, help="Путь к JSON файлу с транзакциями")
    parser.add_argument("--rps", type=int, default=100, help="Запросов в секунду")
    parser.add_argument("--target", type=str, default=TARGET_URL, help="URL receiver'а")
    args = parser.parse_args()
    
    # Если файл не указан, ищем последний
    input_file = args.input
    if input_file is None:
        input_file = find_latest_dataset()
        if input_file is None:
            print("❌ Датасет не найден!")
            print("\nСначала сгенерируйте транзакции:")
            print("  python generate_transactions.py --transactions 10000")
            return
        print(f"Используется последний датасет: {input_file}")
    
    simulator = BankSimulator(args.target)
    
    if not simulator.load_dataset(input_file):
        return
    
    await simulator.run_simulation(rps=args.rps)


if __name__ == "__main__":
    asyncio.run(main())
