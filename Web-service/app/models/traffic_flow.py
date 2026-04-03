from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class DistributionType(str, Enum):
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    PARETO = "pareto"


class TrafficType(str, Enum):
    BACKGROUND = "background"
    ANOMALOUS = "anomalous"


class BackgroundTrafficParams(BaseModel):
    A: float = Field(..., gt=0, description="Amplitude of peak")
    t_m: float = Field(..., description="Time of maximum")
    sigma: float = Field(..., gt=0, description="Standard deviation")

    @field_validator("A")
    @classmethod
    def validate_amplitude(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Amplitude A must be positive, got {v}")
        return v

    @field_validator("sigma")
    @classmethod
    def validate_sigma(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Standard deviation sigma must be positive, got {v}")
        return v


class DistributionParams(BaseModel):
    mean: Optional[float] = None
    variance: Optional[float] = None
    rate: Optional[float] = Field(None, gt=0, description="Lambda for exponential/poisson")
    shape: Optional[float] = Field(None, gt=0, description="Alpha for pareto")
    scale: Optional[float] = Field(None, gt=0, description="x_m for pareto")


class AnomalousTrafficParams(BaseModel):
    distribution: DistributionType
    total_volume: float = Field(..., gt=0, description="N_attack total volume")
    start_time: float = Field(..., description="Attack start time")
    duration: float = Field(..., gt=0, description="Attack duration")
    params: DistributionParams

    @field_validator("total_volume")
    @classmethod
    def validate_volume(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Total volume must be positive, got {v}")
        return v

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Duration must be positive, got {v}")
        return v



class TrafficFlowConfig(BaseModel):
    background: BackgroundTrafficParams
    anomalous: Optional[AnomalousTrafficParams] = None
    time_step: float = Field(default=1.0, gt=0)


class LabeledTransaction(BaseModel):
    transaction_id: str
    timestamp: float
    traffic_type: TrafficType
    distribution: str
    payload: Dict[str, Any] = {}


class TrafficTimeSeries(BaseModel):
    timestamps: List[float]
    n_bg: List[float]
    n_anom: List[float]
    n_total: List[float]
    metadata: Dict[str, Any]

    @property
    def background_count(self) -> int:
        return int(sum(self.n_bg))

    @property
    def anomalous_count(self) -> int:
        return int(sum(self.n_anom))

    @property
    def total_count(self) -> int:
        return int(sum(self.n_total))
