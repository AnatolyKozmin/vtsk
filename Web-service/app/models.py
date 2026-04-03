from sqlalchemy import Column, BigInteger, Integer, String, Text, DateTime, Float, Boolean, Index
from datetime import datetime
from app.database import Base

def utcnow():
    return datetime.utcnow()

class TrafficRequest(Base):
    __tablename__ = "traffic_requests"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(50), unique=True, nullable=False, index=True)
    batch_id = Column(String(50), index=True)
    attack_type = Column(String(30), default="normal")
    payload_size = Column(Integer, default=0)
    sent_at = Column(DateTime, nullable=False)
    source_ip = Column(String(45))
    target_endpoint = Column(String(255))
    http_method = Column(String(10), default="POST")
    headers_count = Column(Integer, default=0)
    is_malicious = Column(Boolean, default=False)
    malicious_pattern = Column(String(50))
    __table_args__ = (Index('idx_batch_sent', 'batch_id', 'sent_at'),)

class TrafficResponse(Base):
    __tablename__ = "traffic_responses"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(50), nullable=False, index=True)
    batch_id = Column(String(50), index=True)
    received_at = Column(DateTime, nullable=False)
    response_time_ms = Column(Float)
    status_code = Column(Integer)
    was_blocked = Column(Boolean, default=False)
    blocked_by = Column(String(50))
    source_ip = Column(String(45))
    passed_through = Column(Boolean, default=True)
    error_message = Column(Text)
    __table_args__ = (Index('idx_response_batch', 'batch_id', 'received_at'),)

class TestSession(Base):
    __tablename__ = "test_sessions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(255))
    attack_type = Column(String(30))
    started_at = Column(DateTime, default=utcnow)
    ended_at = Column(DateTime)
    total_requests = Column(Integer, default=0)
    requests_sent = Column(Integer, default=0)
    requests_received = Column(Integer, default=0)
    requests_blocked = Column(Integer, default=0)
    avg_response_time = Column(Float)
    min_response_time = Column(Float)
    max_response_time = Column(Float)
    throughput_rps = Column(Float)
    status = Column(String(20), default="running")

class BlockedRequest(Base):
    __tablename__ = "blocked_requests"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(50), nullable=False, index=True)
    session_id = Column(String(50), index=True)
    blocked_at = Column(DateTime, default=utcnow)
    blocked_by = Column(String(50))
    block_reason = Column(Text)
    source_ip = Column(String(45))
    attack_signature = Column(String(100))

class LatencyMetric(Base):
    __tablename__ = "latency_metrics"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    timestamp = Column(DateTime, default=utcnow)
    interval_seconds = Column(Integer, default=1)
    requests_count = Column(Integer, default=0)
    avg_latency_ms = Column(Float)
    p50_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    errors_count = Column(Integer, default=0)
    blocked_count = Column(Integer, default=0)

class ProtectionEvent(Base):
    __tablename__ = "protection_events"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    event_type = Column(String(50))
    source = Column(String(50))
    timestamp = Column(DateTime, default=utcnow)
    details = Column(Text)
    severity = Column(String(20))
    source_ip = Column(String(45))
    action_taken = Column(String(50))
