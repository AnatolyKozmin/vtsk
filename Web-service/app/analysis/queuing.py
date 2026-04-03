import numpy as np
from scipy import stats
from scipy.special import factorial
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import math

class TransactionState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

@dataclass
class QueueingMetrics:
    rho: float
    p_wait: float
    e_wait: float
    p_block: float
    e_queue_length: float
    e_system_length: float
    throughput: float
    d_loss: float

class QueuingTheoryAnalyzer:
    def __init__(self, num_servers: int = 10, queue_capacity: int = 1000, service_rate: float = 100.0):
        self.c = num_servers
        self.K = queue_capacity
        self.mu = service_rate
    
    def calculate_utilization(self, lambda_t: float) -> float:
        return lambda_t / (self.c * self.mu)
    
    def erlang_c(self, a: float) -> float:
        if a <= 0:
            return 0.0
        c = self.c
        sum_terms = sum((a ** n) / factorial(n) for n in range(c))
        last_term = (a ** c) / factorial(c) * (c / (c - a)) if a < c else float('inf')
        if sum_terms + last_term == 0:
            return 0.0
        return last_term / (sum_terms + last_term)
    
    def erlang_b(self, a: float) -> float:
        if a <= 0:
            return 0.0
        c = self.c
        numerator = (a ** c) / factorial(c)
        denominator = sum((a ** n) / factorial(n) for n in range(c + 1))
        return numerator / denominator if denominator > 0 else 0.0
    
    def pollaczek_khinchin(self, lambda_t: float, service_variance: float = None) -> float:
        if service_variance is None:
            service_variance = 1 / (self.mu ** 2)
        rho = self.calculate_utilization(lambda_t)
        if rho >= 1:
            return float('inf')
        cs_squared = service_variance * (self.mu ** 2)
        e_wait = (rho / (1 - rho)) * ((1 + cs_squared) / 2) * (1 / self.mu)
        return e_wait
    
    def calculate_p_block_ggck(self, lambda_t: float) -> float:
        a = lambda_t / self.mu
        if a <= 0:
            return 0.0
        rho = self.calculate_utilization(lambda_t)
        if rho >= 1:
            p0_inv = sum((a ** n) / factorial(n) for n in range(self.c))
            if rho < 1:
                p0_inv += ((a ** self.c) / factorial(self.c)) * (1 - rho ** (self.K - self.c + 1)) / (1 - rho)
            else:
                p0_inv += ((a ** self.c) / factorial(self.c)) * (self.K - self.c + 1)
            p0 = 1 / p0_inv if p0_inv > 0 else 0
            if rho < 1:
                p_block = p0 * ((a ** self.c) / factorial(self.c)) * (rho ** (self.K - self.c)) / (1 - rho)
            else:
                p_block = p0 * ((a ** self.c) / factorial(self.c))
            return min(p_block, 1.0)
        return self.erlang_b(a) * 0.1

    def calculate_d_loss(self, p_block_series: List[float], n_tot_series: List[float], dt: float = 1.0) -> float:
        if not p_block_series or not n_tot_series:
            return 0.0
        numerator = sum(p * n * dt for p, n in zip(p_block_series, n_tot_series))
        denominator = sum(n * dt for n in n_tot_series)
        return numerator / denominator if denominator > 0 else 0.0
    
    def calculate_e_t_fail(self, lambda_t: float, threshold_rho: float = 0.95) -> float:
        rho = self.calculate_utilization(lambda_t)
        if rho >= threshold_rho:
            return 0.0
        margin = threshold_rho - rho
        lambda_margin = margin * self.c * self.mu
        if lambda_margin <= 0:
            return float('inf')
        return 1 / lambda_margin
    
    def analyze_system(self, lambda_t: float) -> QueueingMetrics:
        rho = self.calculate_utilization(lambda_t)
        a = lambda_t / self.mu
        p_wait = self.erlang_c(a) if rho < 1 else 1.0
        e_wait = self.pollaczek_khinchin(lambda_t)
        p_block = self.calculate_p_block_ggck(lambda_t)
        if rho < 1:
            e_queue = (rho ** 2) / (1 - rho) if rho < 1 else self.K
            e_system = e_queue + rho * self.c
        else:
            e_queue = self.K
            e_system = self.K + self.c
        throughput = lambda_t * (1 - p_block)
        return QueueingMetrics(
            rho=rho, p_wait=p_wait, e_wait=e_wait, p_block=p_block,
            e_queue_length=e_queue, e_system_length=e_system,
            throughput=throughput, d_loss=p_block
        )

class PaymentSystemMarkov:
    def __init__(self):
        self.states = [TransactionState.QUEUED, TransactionState.PROCESSING, 
                       TransactionState.COMPLETED, TransactionState.REJECTED, TransactionState.TIMEOUT]
        self.transition_counts = np.zeros((5, 5))
        self.transition_matrix = np.zeros((5, 5))
    
    def _state_idx(self, state: TransactionState) -> int:
        return self.states.index(state)
    
    def record_transition(self, from_state: TransactionState, to_state: TransactionState):
        i, j = self._state_idx(from_state), self._state_idx(to_state)
        self.transition_counts[i, j] += 1
        row_sum = self.transition_counts[i].sum()
        if row_sum > 0:
            self.transition_matrix[i] = self.transition_counts[i] / row_sum
    
    def get_stationary_distribution(self) -> Dict[TransactionState, float]:
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            idx = np.argmin(np.abs(eigenvalues - 1))
            stationary = np.real(eigenvectors[:, idx])
            stationary = stationary / stationary.sum()
            return {self.states[i]: float(stationary[i]) for i in range(len(self.states))}
        except:
            return {s: 0.2 for s in self.states}
    
    def get_completion_probability(self, steps: int = 10) -> float:
        try:
            matrix_power = np.linalg.matrix_power(self.transition_matrix, steps)
            completed_idx = self._state_idx(TransactionState.COMPLETED)
            queued_idx = self._state_idx(TransactionState.QUEUED)
            return float(matrix_power[queued_idx, completed_idx])
        except:
            return 0.0

class PaymentAnomalyType:
    NORMAL = 0
    MICRO_TRANSACTION = 1
    HIGH_AMOUNT = 2
    UNUSUAL_TIME = 3
    GEO_ANOMALY = 4
    VELOCITY_SPIKE = 5
    INTERRUPTED = 6
    HIGH_VOLATILITY = 7
    
    @classmethod
    def to_russian(cls, code: int) -> str:
        names = {
            cls.NORMAL: "Норма",
            cls.MICRO_TRANSACTION: "Микротранзакция",
            cls.HIGH_AMOUNT: "Высокая сумма",
            cls.UNUSUAL_TIME: "Необычное время",
            cls.GEO_ANOMALY: "Гео-аномалия",
            cls.VELOCITY_SPIKE: "Всплеск частоты",
            cls.INTERRUPTED: "Прерванный платёж",
            cls.HIGH_VOLATILITY: "Высокая волатильность",
        }
        return names.get(code, "Неизвестно")
