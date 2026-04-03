from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class AttackType(str, Enum):
    NORMAL = "normal"
    FLOOD = "flood"
    SLOWLORIS = "slowloris"
    BURST = "burst"
    AMPLIFICATION = "amplification"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    RANDOM = "random"

class TrafficPayload(BaseModel):
    request_id: str
    batch_id: str
    attack_type: AttackType = AttackType.NORMAL
    timestamp: str
    payload: Dict[str, Any] = {}
    payload_size: int = 0
    is_malicious: bool = False
    malicious_pattern: Optional[str] = None
    headers: Dict[str, str] = {}

class TrafficBatch(BaseModel):
    batch_id: str
    session_id: str
    requests: List[TrafficPayload]
    attack_type: AttackType = AttackType.NORMAL
    sent_at: str

class RequestResult(BaseModel):
    request_id: str
    received: bool
    response_time_ms: float
    was_blocked: bool = False
    blocked_by: Optional[str] = None
    status_code: int = 200
    error: Optional[str] = None

class BatchResult(BaseModel):
    batch_id: str
    session_id: str
    total_requests: int
    received_count: int
    blocked_count: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    timestamp: str
    results: List[RequestResult]

class SessionConfig(BaseModel):
    name: str = "Test Session"
    attack_type: AttackType = AttackType.NORMAL
    total_requests: int = 1000
    requests_per_second: int = 100
    duration_seconds: Optional[int] = None
    burst_size: Optional[int] = None
    burst_interval_ms: Optional[int] = None
    payload_size_bytes: int = 1024
    include_malicious: bool = False
    malicious_ratio: float = 0.1

class SessionStatus(BaseModel):
    session_id: str
    name: str
    status: str
    attack_type: str
    started_at: str
    ended_at: Optional[str] = None
    total_requests: int
    requests_sent: int
    requests_received: int
    requests_blocked: int
    block_rate: float
    avg_response_time_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    progress_percent: float

class LatencyStats(BaseModel):
    session_id: str
    total_requests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_dev_ms: float

class ProtectionStats(BaseModel):
    session_id: str
    total_blocked: int
    blocked_by_firewall: int
    blocked_by_waf: int
    blocked_by_ids: int
    blocked_by_rate_limit: int
    block_rate_percent: float
    detection_rate_percent: float
    false_positive_rate: float

class TestReport(BaseModel):
    session_id: str
    name: str
    attack_type: str
    duration_seconds: float
    total_requests: int
    requests_received: int
    requests_blocked: int
    latency: LatencyStats
    protection: ProtectionStats
    timeline: List[Dict[str, Any]]
    recommendations: List[str]

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    uptime_seconds: float
    requests_processed: int
