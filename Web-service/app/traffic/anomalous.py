import math
from typing import List, Tuple
import numpy as np
from app.models.traffic_flow import AnomalousTrafficParams, DistributionType, TrafficType


class AnomalousTrafficGenerator:
    def __init__(self, params: AnomalousTrafficParams):
        self.params = params
        self._normalization_factor: float = 1.0
        self._compute_normalization()

    def _compute_normalization(self) -> None:
        dt = 0.1
        start = self.params.start_time
        end = self.params.start_time + self.params.duration
        raw_sum = 0.0
        t = start
        while t <= end:
            raw_sum += self._raw_compute(t) * dt
            t += dt
        if raw_sum > 0:
            self._normalization_factor = self.params.total_volume / raw_sum
        else:
            self._normalization_factor = 1.0

    def _raw_compute(self, t: float) -> float:
        start = self.params.start_time
        end = self.params.start_time + self.params.duration
        if t < start or t > end:
            return 0.0
        rel_t = (t - start) / self.params.duration
        dist = self.params.distribution
        p = self.params.params
        if dist == DistributionType.NORMAL:
            mean = p.mean if p.mean is not None else 0.5
            var = p.variance if p.variance is not None else 0.1
            std = math.sqrt(var) if var > 0 else 0.1
            return math.exp(-((rel_t - mean) ** 2) / (2 * std ** 2))
        elif dist == DistributionType.EXPONENTIAL:
            rate = p.rate if p.rate is not None else 2.0
            return rate * math.exp(-rate * rel_t)
        elif dist == DistributionType.POISSON:
            lam = p.rate if p.rate is not None else 5.0
            k = int(rel_t * 10)
            return (lam ** k) * math.exp(-lam) / math.factorial(min(k, 20))
        elif dist == DistributionType.PARETO:
            alpha = p.shape if p.shape is not None else 2.0
            x_m = p.scale if p.scale is not None else 0.1
            x = max(rel_t, x_m)
            return alpha * (x_m ** alpha) / (x ** (alpha + 1))
        return 0.0

    def compute(self, t: float) -> float:
        return self._raw_compute(t) * self._normalization_factor


    def compute_series(self, start_time: float, end_time: float, dt: float) -> List[Tuple[float, float]]:
        result = []
        t = start_time
        while t <= end_time:
            result.append((t, self.compute(t)))
            t += dt
        return result

    def get_total_volume(self, dt: float = 0.1) -> float:
        start = self.params.start_time
        end = self.params.start_time + self.params.duration
        total = 0.0
        t = start
        while t <= end:
            total += self.compute(t) * dt
            t += dt
        return total

    @property
    def traffic_type(self) -> TrafficType:
        return TrafficType.ANOMALOUS

    @property
    def distribution_name(self) -> str:
        return self.params.distribution.value
