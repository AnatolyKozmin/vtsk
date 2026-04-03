from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SLAComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"


class QueueingSystemConfig(BaseModel):
    c: int = Field(..., gt=0, description="Number of servers")
    K: int = Field(..., ge=0, description="Queue capacity")
    mu: float = Field(..., gt=0, description="Service rate")
    sla_epsilon: float = Field(default=0.01, ge=0, le=1, description="P_block threshold")
    sla_delta: float = Field(default=0.05, ge=0, le=1, description="D_loss threshold")

    @field_validator("c")
    @classmethod
    def validate_servers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Server count c must be positive, got {v}")
        return v

    @field_validator("K")
    @classmethod
    def validate_capacity(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Queue capacity K must be non-negative, got {v}")
        return v

    @field_validator("mu")
    @classmethod
    def validate_service_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Service rate mu must be positive")
        return v


class UtilizationResult(BaseModel):
    timestamps: List[float]
    instantaneous: List[float]
    moving_average: List[float]
    overload_periods: List[Tuple[float, float]]
    max_utilization: float
    avg_utilization: float


class DLossResult(BaseModel):
    d_loss: float
    numerator: float
    denominator: float
    dt: float
    sla_compliant: bool
    violation_magnitude: Optional[float] = None



class BlockingEvent(BaseModel):
    timestamp: float
    queue_length: int
    arrival_rate: float
    utilization: float


class QueueingMetricsSnapshot(BaseModel):
    timestamp: float
    rho: float
    p_block: float
    queue_length: float
    throughput: float
    e_wait: float


class QueueingAnalysisResult(BaseModel):
    config: QueueingSystemConfig
    utilization: UtilizationResult
    p_block_series: List[Tuple[float, float]]
    d_loss: DLossResult
    e_t_fail: float
    blocking_events: List[BlockingEvent]
    metrics_timeline: List[QueueingMetricsSnapshot]


class SLAComplianceResult(BaseModel):
    metric_name: str
    actual_value: float
    threshold: float
    status: SLAComplianceStatus
    violation_severity_percent: Optional[float] = None


class SLAViolation(BaseModel):
    metric_name: str
    actual_value: float
    threshold: float
    severity_percent: float
    start_time: float
    duration: float


class SystemReport(BaseModel):
    report_id: str
    generated_at: datetime
    simulation_params: Dict[str, Any]
    utilization_summary: Dict[str, float]
    p_block_summary: Dict[str, float]
    d_loss: float
    e_t_fail: float
    throughput: float
    avg_queue_length: float
    sla_compliance: List[SLAComplianceResult]
    violations: List[SLAViolation]
    overall_compliant: bool
    recommendations: List[str]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ComparisonReport(BaseModel):
    analytical_metrics: Dict[str, float]
    simulated_metrics: Dict[str, float]
    relative_errors: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    discrepancies: List[str]
    sample_size: int
