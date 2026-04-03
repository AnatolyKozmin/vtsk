import math
from typing import List, Tuple
from app.models.traffic_flow import BackgroundTrafficParams, TrafficType


class BackgroundTrafficGenerator:
    def __init__(self, params: BackgroundTrafficParams):
        self.params = params
        self.validate_params()

    def validate_params(self) -> None:
        if self.params.A <= 0:
            raise ValueError(f"Amplitude A must be positive, got {self.params.A}")
        if self.params.sigma <= 0:
            raise ValueError(f"Standard deviation sigma must be positive, got {self.params.sigma}")

    def compute(self, t: float) -> float:
        A = self.params.A
        t_m = self.params.t_m
        sigma = self.params.sigma
        exponent = -((t - t_m) ** 2) / (2 * sigma ** 2)
        return A * math.exp(exponent)

    def compute_series(self, start_time: float, end_time: float, dt: float) -> List[Tuple[float, float]]:
        result = []
        t = start_time
        while t <= end_time:
            result.append((t, self.compute(t)))
            t += dt
        return result

    @property
    def traffic_type(self) -> TrafficType:
        return TrafficType.BACKGROUND

    @property
    def distribution_name(self) -> str:
        return "gaussian"
