import numpy as np
from scipy import stats
from scipy.special import factorial
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

class DistributionType(str, Enum):
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    PARETO = "pareto"
    WEIBULL = "weibull"

@dataclass
class TrafficModel:
    lambda_intensity: float
    variance: float
    mean: float
    std_dev: float
    distribution: DistributionType
    confidence_interval: Tuple[float, float]
    anomaly_threshold: float

@dataclass
class MarkovState:
    name: str
    probability: float
    transitions: Dict[str, float]

class StatisticalAnalyzer:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_poisson_parameters(self, arrivals: List[float], time_window: float) -> Dict:
        if not arrivals:
            return {"lambda": 0, "variance": 0, "mean": 0}
        
        n = len(arrivals)
        lambda_mle = n / time_window
        variance = lambda_mle
        mean = lambda_mle
        
        z = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = lambda_mle - z * np.sqrt(lambda_mle / time_window)
        ci_upper = lambda_mle + z * np.sqrt(lambda_mle / time_window)
        
        return {
            "lambda": lambda_mle,
            "variance": variance,
            "mean": mean,
            "std_dev": np.sqrt(variance),
            "confidence_interval": (max(0, ci_lower), ci_upper),
            "distribution": DistributionType.POISSON
        }
    
    def calculate_exponential_parameters(self, inter_arrival_times: List[float]) -> Dict:
        if not inter_arrival_times or len(inter_arrival_times) < 2:
            return {"lambda": 0, "mean": 0, "variance": 0}
        
        mean_iat = np.mean(inter_arrival_times)
        lambda_rate = 1 / mean_iat if mean_iat > 0 else 0
        variance = 1 / (lambda_rate ** 2) if lambda_rate > 0 else 0
        
        n = len(inter_arrival_times)
        chi2_lower = stats.chi2.ppf(self.alpha / 2, 2 * n)
        chi2_upper = stats.chi2.ppf(1 - self.alpha / 2, 2 * n)
        
        ci_lower = 2 * n * lambda_rate / chi2_upper
        ci_upper = 2 * n * lambda_rate / chi2_lower
        
        return {
            "lambda": lambda_rate,
            "mean": mean_iat,
            "variance": variance,
            "std_dev": np.sqrt(variance),
            "confidence_interval": (ci_lower, ci_upper),
            "distribution": DistributionType.EXPONENTIAL
        }
    
    def fit_distribution(self, data: List[float]) -> Tuple[DistributionType, Dict]:
        if not data or len(data) < 10:
            return DistributionType.NORMAL, {}
        
        data_array = np.array(data)
        
        distributions = {
            DistributionType.NORMAL: stats.norm,
            DistributionType.EXPONENTIAL: stats.expon,
            DistributionType.POISSON: None,
            DistributionType.PARETO: stats.pareto,
            DistributionType.WEIBULL: stats.weibull_min
        }
        
        best_dist = DistributionType.NORMAL
        best_ks = float('inf')
        best_params = {}
        
        for dist_type, dist in distributions.items():
            if dist is None:
                continue
            try:
                params = dist.fit(data_array)
                ks_stat, _ = stats.kstest(data_array, dist.cdf, args=params)
                if ks_stat < best_ks:
                    best_ks = ks_stat
                    best_dist = dist_type
                    best_params = {"params": params, "ks_statistic": ks_stat}
            except:
                continue
        
        return best_dist, best_params
    
    def hypothesis_test_ddos(self, baseline_data: List[float], current_data: List[float]) -> Dict:
        if len(baseline_data) < 5 or len(current_data) < 5:
            return {"is_anomaly": False, "p_value": 1.0, "test": "insufficient_data"}
        
        t_stat, p_value_ttest = stats.ttest_ind(baseline_data, current_data)
        
        u_stat, p_value_mann = stats.mannwhitneyu(baseline_data, current_data, alternative='two-sided')
        
        ks_stat, p_value_ks = stats.ks_2samp(baseline_data, current_data)
        
        combined_p = 3 / (1/p_value_ttest + 1/p_value_mann + 1/p_value_ks) if all(p > 0 for p in [p_value_ttest, p_value_mann, p_value_ks]) else 1.0
        
        return {
            "is_anomaly": combined_p < self.alpha,
            "t_test": {"statistic": t_stat, "p_value": p_value_ttest},
            "mann_whitney": {"statistic": u_stat, "p_value": p_value_mann},
            "ks_test": {"statistic": ks_stat, "p_value": p_value_ks},
            "combined_p_value": combined_p,
            "significance_level": self.alpha
        }
    
    def calculate_anomaly_threshold(self, data: List[float], method: str = "iqr") -> float:
        if not data:
            return 0
        
        data_array = np.array(data)
        
        if method == "iqr":
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            return q3 + 1.5 * iqr
        elif method == "zscore":
            mean = np.mean(data_array)
            std = np.std(data_array)
            z_critical = stats.norm.ppf(1 - self.alpha)
            return mean + z_critical * std
        elif method == "mad":
            median = np.median(data_array)
            mad = np.median(np.abs(data_array - median))
            return median + 3 * 1.4826 * mad
        else:
            return np.percentile(data_array, 95)
    
    def build_traffic_model(self, timestamps: List[float], values: List[float]) -> TrafficModel:
        if not timestamps or not values:
            return TrafficModel(0, 0, 0, 0, DistributionType.NORMAL, (0, 0), 0)
        
        inter_arrivals = np.diff(sorted(timestamps)).tolist() if len(timestamps) > 1 else [1.0]
        
        exp_params = self.calculate_exponential_parameters(inter_arrivals)
        
        dist_type, _ = self.fit_distribution(values)
        
        threshold = self.calculate_anomaly_threshold(values)
        
        return TrafficModel(
            lambda_intensity=exp_params["lambda"],
            variance=np.var(values),
            mean=np.mean(values),
            std_dev=np.std(values),
            distribution=dist_type,
            confidence_interval=exp_params["confidence_interval"],
            anomaly_threshold=threshold
        )

class MarkovChainAnalyzer:
    def __init__(self):
        self.states = ["normal", "suspicious", "attack", "blocked"]
        self.transition_matrix = np.zeros((4, 4))
        self.state_counts = {s: 0 for s in self.states}
        self.transition_counts = np.zeros((4, 4))
    
    def _state_index(self, state: str) -> int:
        return self.states.index(state) if state in self.states else 0
    
    def update_transition(self, from_state: str, to_state: str):
        i, j = self._state_index(from_state), self._state_index(to_state)
        self.transition_counts[i, j] += 1
        self.state_counts[from_state] = self.state_counts.get(from_state, 0) + 1
        
        row_sum = self.transition_counts[i].sum()
        if row_sum > 0:
            self.transition_matrix[i] = self.transition_counts[i] / row_sum
    
    def get_stationary_distribution(self) -> Dict[str, float]:
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            stationary_idx = np.argmin(np.abs(eigenvalues - 1))
            stationary = np.real(eigenvectors[:, stationary_idx])
            stationary = stationary / stationary.sum()
            return {self.states[i]: float(stationary[i]) for i in range(len(self.states))}
        except:
            return {s: 0.25 for s in self.states}
    
    def predict_next_state(self, current_state: str) -> Tuple[str, float]:
        i = self._state_index(current_state)
        probs = self.transition_matrix[i]
        if probs.sum() == 0:
            return "normal", 0.25
        next_idx = np.argmax(probs)
        return self.states[next_idx], float(probs[next_idx])
    
    def get_attack_probability(self, steps: int = 1) -> float:
        try:
            matrix_power = np.linalg.matrix_power(self.transition_matrix, steps)
            attack_idx = self._state_index("attack")
            return float(matrix_power[0, attack_idx])
        except:
            return 0.0
    
    def get_mean_time_to_attack(self) -> float:
        try:
            attack_idx = self._state_index("attack")
            Q = np.delete(np.delete(self.transition_matrix, attack_idx, 0), attack_idx, 1)
            I = np.eye(Q.shape[0])
            N = np.linalg.inv(I - Q)
            return float(N[0].sum())
        except:
            return float('inf')

class AnomalyDetector:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.baseline_data: List[float] = []
        self.analyzer = StatisticalAnalyzer()
        self.markov = MarkovChainAnalyzer()
        self.current_state = "normal"
    
    def update_baseline(self, value: float):
        self.baseline_data.append(value)
        if len(self.baseline_data) > self.window_size * 10:
            self.baseline_data = self.baseline_data[-self.window_size * 10:]
    
    def detect_anomaly(self, current_values: List[float]) -> Dict:
        if len(self.baseline_data) < self.window_size:
            return {"is_anomaly": False, "confidence": 0, "reason": "insufficient_baseline"}
        
        baseline_window = self.baseline_data[-self.window_size:]
        
        test_result = self.analyzer.hypothesis_test_ddos(baseline_window, current_values)
        
        threshold = self.analyzer.calculate_anomaly_threshold(baseline_window)
        current_mean = np.mean(current_values)
        threshold_exceeded = current_mean > threshold
        
        baseline_mean = np.mean(baseline_window)
        baseline_std = np.std(baseline_window)
        z_score = (current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        is_anomaly = test_result["is_anomaly"] or threshold_exceeded or abs(z_score) > 3
        
        new_state = "attack" if is_anomaly else "normal"
        if abs(z_score) > 2 and not is_anomaly:
            new_state = "suspicious"
        
        self.markov.update_transition(self.current_state, new_state)
        self.current_state = new_state
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": 1 - test_result["combined_p_value"],
            "z_score": z_score,
            "threshold": threshold,
            "current_mean": current_mean,
            "baseline_mean": baseline_mean,
            "statistical_test": test_result,
            "markov_state": new_state,
            "attack_probability": self.markov.get_attack_probability(5)
        }
