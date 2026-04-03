import uuid
import json
from typing import Tuple, List, Optional, AsyncGenerator, Dict, Any
from app.models.traffic_flow import (
    TrafficFlowConfig,
    TrafficTimeSeries,
    LabeledTransaction,
    TrafficType,
)
from app.traffic.background import BackgroundTrafficGenerator
from app.traffic.anomalous import AnomalousTrafficGenerator


class TrafficFlowGenerator:
    def __init__(self, config: TrafficFlowConfig):
        self.config = config
        self._bg_generator = BackgroundTrafficGenerator(config.background)
        self._anom_generator: Optional[AnomalousTrafficGenerator] = None
        if config.anomalous:
            self._anom_generator = AnomalousTrafficGenerator(config.anomalous)
        self._bg_count = 0
        self._anom_count = 0

    def generate_background(self, t: float) -> float:
        return self._bg_generator.compute(t)

    def generate_anomalous(self, t: float) -> float:
        if self._anom_generator is None:
            return 0.0
        return self._anom_generator.compute(t)

    def generate_combined(self, t: float) -> Tuple[float, float, float]:
        n_bg = self.generate_background(t)
        n_anom = self.generate_anomalous(t)
        n_total = n_bg + n_anom
        return (n_bg, n_anom, n_total)

    def generate_time_series(
        self, start_time: float, end_time: float, dt: Optional[float] = None
    ) -> TrafficTimeSeries:
        if dt is None:
            dt = self.config.time_step
        timestamps = []
        n_bg_list = []
        n_anom_list = []
        n_total_list = []
        t = start_time
        while t <= end_time:
            n_bg, n_anom, n_total = self.generate_combined(t)
            timestamps.append(t)
            n_bg_list.append(n_bg)
            n_anom_list.append(n_anom)
            n_total_list.append(n_total)
            t += dt

        metadata = {
            "background_params": {
                "A": self.config.background.A,
                "t_m": self.config.background.t_m,
                "sigma": self.config.background.sigma,
            },
            "time_step": dt,
            "start_time": start_time,
            "end_time": end_time,
        }
        if self.config.anomalous:
            metadata["anomalous_params"] = {
                "distribution": self.config.anomalous.distribution.value,
                "total_volume": self.config.anomalous.total_volume,
                "start_time": self.config.anomalous.start_time,
                "duration": self.config.anomalous.duration,
            }
        return TrafficTimeSeries(
            timestamps=timestamps,
            n_bg=n_bg_list,
            n_anom=n_anom_list,
            n_total=n_total_list,
            metadata=metadata,
        )

    async def generate_transactions(
        self, time_range: Tuple[float, float], dt: Optional[float] = None
    ) -> AsyncGenerator[LabeledTransaction, None]:
        if dt is None:
            dt = self.config.time_step
        start_time, end_time = time_range
        t = start_time
        while t <= end_time:
            n_bg, n_anom, _ = self.generate_combined(t)
            bg_count = int(n_bg * dt)
            anom_count = int(n_anom * dt)
            for _ in range(bg_count):
                self._bg_count += 1
                yield LabeledTransaction(
                    transaction_id=str(uuid.uuid4()),
                    timestamp=t,
                    traffic_type=TrafficType.BACKGROUND,
                    distribution="gaussian",
                )
            for _ in range(anom_count):
                self._anom_count += 1
                dist_name = "none"
                if self._anom_generator:
                    dist_name = self._anom_generator.distribution_name
                yield LabeledTransaction(
                    transaction_id=str(uuid.uuid4()),
                    timestamp=t,
                    traffic_type=TrafficType.ANOMALOUS,
                    distribution=dist_name,
                )
            t += dt

    @property
    def background_count(self) -> int:
        return self._bg_count

    @property
    def anomalous_count(self) -> int:
        return self._anom_count

    def reset_counters(self) -> None:
        self._bg_count = 0
        self._anom_count = 0


    @staticmethod
    def serialize(data: TrafficTimeSeries) -> str:
        return data.model_dump_json()

    @staticmethod
    def deserialize(json_str: str) -> TrafficTimeSeries:
        return TrafficTimeSeries.model_validate_json(json_str)
