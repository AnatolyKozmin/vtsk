import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import threading

@dataclass
class RequestMetric:
    request_id: str
    sent_at: float
    received_at: Optional[float] = None
    response_time_ms: Optional[float] = None
    was_blocked: bool = False
    blocked_by: Optional[str] = None
    status_code: int = 0
    is_malicious: bool = False
    attack_type: str = "normal"
    error: Optional[str] = None

@dataclass
class IntervalStats:
    timestamp: datetime
    requests_sent: int = 0
    requests_received: int = 0
    requests_blocked: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: int = 0
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

class MetricsCollector:
    def __init__(self, session_id: str, interval_seconds: int = 1):
        self.session_id = session_id
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._metrics: Dict[str, RequestMetric] = {}
        self._intervals: deque = deque(maxlen=3600)
        self._current_interval: Optional[IntervalStats] = None
        self._interval_start: Optional[datetime] = None
        self._started_at: Optional[datetime] = None
        self._total_sent = 0
        self._total_received = 0
        self._total_blocked = 0
        self._blocked_by_source: Dict[str, int] = {}
        self._attack_stats: Dict[str, Dict[str, int]] = {}
    
    def start(self):
        self._started_at = datetime.now(timezone.utc)
        self._interval_start = self._started_at
        self._current_interval = IntervalStats(timestamp=self._started_at)
    
    def _rotate_interval(self):
        now = datetime.now(timezone.utc)
        if self._interval_start and (now - self._interval_start).total_seconds() >= self.interval_seconds:
            if self._current_interval:
                self._intervals.append(self._current_interval)
            self._interval_start = now
            self._current_interval = IntervalStats(timestamp=now)
    
    def record_sent(self, request_id: str, is_malicious: bool = False, attack_type: str = "normal"):
        with self._lock:
            self._rotate_interval()
            self._metrics[request_id] = RequestMetric(
                request_id=request_id,
                sent_at=datetime.now(timezone.utc).timestamp(),
                is_malicious=is_malicious,
                attack_type=attack_type
            )
            self._total_sent += 1
            if self._current_interval:
                self._current_interval.requests_sent += 1
            if attack_type not in self._attack_stats:
                self._attack_stats[attack_type] = {"sent": 0, "blocked": 0, "passed": 0}
            self._attack_stats[attack_type]["sent"] += 1
    
    def record_received(self, request_id: str, response_time_ms: float, status_code: int = 200,
                       was_blocked: bool = False, blocked_by: Optional[str] = None, error: Optional[str] = None):
        with self._lock:
            self._rotate_interval()
            attack_type = "normal"
            if request_id in self._metrics:
                metric = self._metrics[request_id]
                metric.received_at = datetime.now(timezone.utc).timestamp()
                metric.response_time_ms = response_time_ms
                metric.status_code = status_code
                metric.was_blocked = was_blocked
                metric.blocked_by = blocked_by
                metric.error = error
                attack_type = metric.attack_type
            
            self._total_received += 1
            if was_blocked:
                self._total_blocked += 1
                if blocked_by:
                    self._blocked_by_source[blocked_by] = self._blocked_by_source.get(blocked_by, 0) + 1
                if attack_type in self._attack_stats:
                    self._attack_stats[attack_type]["blocked"] += 1
            else:
                if attack_type in self._attack_stats:
                    self._attack_stats[attack_type]["passed"] += 1
            
            if self._current_interval:
                self._current_interval.requests_received += 1
                if was_blocked:
                    self._current_interval.requests_blocked += 1
                if response_time_ms:
                    self._current_interval.latencies.append(response_time_ms)
                if error:
                    self._current_interval.errors += 1
    
    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            all_latencies = [m.response_time_ms for m in self._metrics.values() if m.response_time_ms]
            duration = (datetime.now(timezone.utc) - self._started_at).total_seconds() if self._started_at else 0
            
            return {
                "session_id": self.session_id,
                "duration_seconds": duration,
                "total_sent": self._total_sent,
                "total_received": self._total_received,
                "total_blocked": self._total_blocked,
                "block_rate": self._total_blocked / self._total_received * 100 if self._total_received > 0 else 0,
                "throughput_rps": self._total_sent / duration if duration > 0 else 0,
                "latency": {
                    "avg_ms": statistics.mean(all_latencies) if all_latencies else 0,
                    "min_ms": min(all_latencies) if all_latencies else 0,
                    "max_ms": max(all_latencies) if all_latencies else 0,
                    "p50_ms": statistics.median(all_latencies) if all_latencies else 0,
                    "p95_ms": sorted(all_latencies)[int(len(all_latencies) * 0.95)] if len(all_latencies) > 1 else 0,
                    "p99_ms": sorted(all_latencies)[int(len(all_latencies) * 0.99)] if len(all_latencies) > 1 else 0,
                    "std_dev_ms": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
                },
                "blocked_by": self._blocked_by_source,
                "attack_stats": self._attack_stats
            }
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{
                "timestamp": i.timestamp.isoformat(),
                "requests_sent": i.requests_sent,
                "requests_received": i.requests_received,
                "requests_blocked": i.requests_blocked,
                "avg_latency_ms": i.avg_latency,
                "p95_latency_ms": i.p95_latency,
                "errors": i.errors
            } for i in self._intervals]
    
    def get_protection_effectiveness(self) -> Dict[str, Any]:
        with self._lock:
            malicious_sent = sum(1 for m in self._metrics.values() if m.is_malicious)
            malicious_blocked = sum(1 for m in self._metrics.values() if m.is_malicious and m.was_blocked)
            normal_sent = sum(1 for m in self._metrics.values() if not m.is_malicious)
            normal_blocked = sum(1 for m in self._metrics.values() if not m.is_malicious and m.was_blocked)
            
            return {
                "malicious_sent": malicious_sent,
                "malicious_blocked": malicious_blocked,
                "malicious_passed": malicious_sent - malicious_blocked,
                "normal_sent": normal_sent,
                "normal_blocked": normal_blocked,
                "detection_rate_percent": malicious_blocked / malicious_sent * 100 if malicious_sent > 0 else 0,
                "false_positive_rate_percent": normal_blocked / normal_sent * 100 if normal_sent > 0 else 0,
                "blocked_by_source": self._blocked_by_source,
                "attack_type_stats": self._attack_stats
            }
