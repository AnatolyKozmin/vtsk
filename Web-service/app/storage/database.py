from sqlalchemy import Column, BigInteger, Integer, String, Text, DateTime, Float, Boolean, Index, JSON, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import select, func, text
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import json

class Base(DeclarativeBase):
    pass

class TestSession(Base):
    __tablename__ = "test_sessions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255))
    attack_type = Column(String(50))
    config = Column(JSON)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime)
    status = Column(String(20), default="running")
    total_requests = Column(Integer, default=0)
    requests_sent = Column(Integer, default=0)
    requests_received = Column(Integer, default=0)
    requests_blocked = Column(Integer, default=0)
    summary_json = Column(JSON)
    
    events = relationship("TrafficEvent", back_populates="session")
    metrics = relationship("IntervalMetric", back_populates="session")

class TrafficEvent(Base):
    __tablename__ = "traffic_events"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey("test_sessions.session_id"), index=True)
    request_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    source_ip = Column(String(45))
    dest_ip = Column(String(45))
    source_port = Column(Integer)
    dest_port = Column(Integer)
    protocol = Column(String(10))
    packet_size = Column(Integer)
    response_time_ms = Column(Float)
    status_code = Column(Integer)
    was_blocked = Column(Boolean, default=False)
    blocked_by = Column(String(50))
    attack_type = Column(String(50))
    is_malicious = Column(Boolean, default=False)
    payload_hash = Column(String(64))
    headers_json = Column(JSON)
    geo_location = Column(String(10))
    
    session = relationship("TestSession", back_populates="events")
    
    __table_args__ = (
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_blocked', 'was_blocked', 'blocked_by'),
        Index('idx_attack_type', 'attack_type', 'is_malicious'),
    )

class IntervalMetric(Base):
    __tablename__ = "interval_metrics"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey("test_sessions.session_id"), index=True)
    interval_start = Column(DateTime, nullable=False, index=True)
    interval_seconds = Column(Integer, default=1)
    requests_count = Column(Integer, default=0)
    blocked_count = Column(Integer, default=0)
    malicious_count = Column(Integer, default=0)
    avg_latency_ms = Column(Float)
    p50_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    min_latency_ms = Column(Float)
    max_latency_ms = Column(Float)
    throughput_rps = Column(Float)
    bytes_sent = Column(BigInteger, default=0)
    bytes_received = Column(BigInteger, default=0)
    unique_sources = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    session = relationship("TestSession", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_interval_session', 'session_id', 'interval_start'),
    )

class StatisticalModel(Base):
    __tablename__ = "statistical_models"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    model_type = Column(String(50))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    parameters_json = Column(JSON)
    lambda_intensity = Column(Float)
    variance = Column(Float)
    mean_value = Column(Float)
    std_dev = Column(Float)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    anomaly_threshold = Column(Float)
    distribution_type = Column(String(30))

class SLAViolation(Base):
    __tablename__ = "sla_violations"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    metric_name = Column(String(50))
    violation_start = Column(DateTime, nullable=False)
    violation_end = Column(DateTime)
    target_value = Column(Float)
    actual_value = Column(Float)
    severity = Column(String(20))
    duration_seconds = Column(Float)
    cost_impact = Column(Float)

class AttackPattern(Base):
    __tablename__ = "attack_patterns"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    attack_type = Column(String(50))
    confidence = Column(Float)
    signature_matched = Column(String(100))
    features_json = Column(JSON)
    source_ips = Column(JSON)
    duration_seconds = Column(Float)
    peak_intensity_rps = Column(Float)

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False, pool_size=20, max_overflow=10)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_session(self) -> AsyncSession:
        return self.async_session()
    
    async def save_test_session(self, session_data: Dict) -> str:
        async with self.async_session() as db:
            session = TestSession(**session_data)
            db.add(session)
            await db.commit()
            return session.session_id
    
    async def save_traffic_events_batch(self, events: List[Dict]):
        async with self.async_session() as db:
            for event_data in events:
                event = TrafficEvent(**event_data)
                db.add(event)
            await db.commit()
    
    async def save_interval_metric(self, metric_data: Dict):
        async with self.async_session() as db:
            metric = IntervalMetric(**metric_data)
            db.add(metric)
            await db.commit()
    
    async def save_statistical_model(self, model_data: Dict):
        async with self.async_session() as db:
            model = StatisticalModel(**model_data)
            db.add(model)
            await db.commit()
    
    async def save_sla_violation(self, violation_data: Dict):
        async with self.async_session() as db:
            violation = SLAViolation(**violation_data)
            db.add(violation)
            await db.commit()
    
    async def save_attack_pattern(self, pattern_data: Dict):
        async with self.async_session() as db:
            pattern = AttackPattern(**pattern_data)
            db.add(pattern)
            await db.commit()
    
    async def get_session_events(self, session_id: str, limit: int = 10000) -> List[Dict]:
        async with self.async_session() as db:
            result = await db.execute(
                select(TrafficEvent)
                .where(TrafficEvent.session_id == session_id)
                .order_by(TrafficEvent.timestamp)
                .limit(limit)
            )
            events = result.scalars().all()
            return [self._event_to_dict(e) for e in events]
    
    async def get_interval_metrics(self, session_id: str) -> List[Dict]:
        async with self.async_session() as db:
            result = await db.execute(
                select(IntervalMetric)
                .where(IntervalMetric.session_id == session_id)
                .order_by(IntervalMetric.interval_start)
            )
            metrics = result.scalars().all()
            return [self._metric_to_dict(m) for m in metrics]
    
    async def get_aggregated_stats(self, session_id: str) -> Dict:
        async with self.async_session() as db:
            result = await db.execute(
                select(
                    func.count(TrafficEvent.id).label("total"),
                    func.sum(func.cast(TrafficEvent.was_blocked, Integer)).label("blocked"),
                    func.sum(func.cast(TrafficEvent.is_malicious, Integer)).label("malicious"),
                    func.avg(TrafficEvent.response_time_ms).label("avg_latency"),
                    func.min(TrafficEvent.response_time_ms).label("min_latency"),
                    func.max(TrafficEvent.response_time_ms).label("max_latency"),
                    func.count(func.distinct(TrafficEvent.source_ip)).label("unique_sources")
                ).where(TrafficEvent.session_id == session_id)
            )
            row = result.first()
            
            return {
                "total_events": row.total or 0,
                "blocked_events": row.blocked or 0,
                "malicious_events": row.malicious or 0,
                "avg_latency_ms": float(row.avg_latency or 0),
                "min_latency_ms": float(row.min_latency or 0),
                "max_latency_ms": float(row.max_latency or 0),
                "unique_sources": row.unique_sources or 0
            }
    
    async def export_for_analysis(self, session_id: str, format: str = "dict") -> Any:
        events = await self.get_session_events(session_id)
        metrics = await self.get_interval_metrics(session_id)
        stats = await self.get_aggregated_stats(session_id)
        
        data = {
            "session_id": session_id,
            "events": events,
            "interval_metrics": metrics,
            "aggregated_stats": stats
        }
        
        if format == "json":
            return json.dumps(data, default=str)
        
        return data
    
    def _event_to_dict(self, event: TrafficEvent) -> Dict:
        return {
            "request_id": event.request_id,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            "source_ip": event.source_ip,
            "response_time_ms": event.response_time_ms,
            "was_blocked": event.was_blocked,
            "blocked_by": event.blocked_by,
            "attack_type": event.attack_type,
            "is_malicious": event.is_malicious,
            "packet_size": event.packet_size
        }
    
    def _metric_to_dict(self, metric: IntervalMetric) -> Dict:
        return {
            "interval_start": metric.interval_start.isoformat() if metric.interval_start else None,
            "requests_count": metric.requests_count,
            "blocked_count": metric.blocked_count,
            "avg_latency_ms": metric.avg_latency_ms,
            "p95_latency_ms": metric.p95_latency_ms,
            "throughput_rps": metric.throughput_rps
        }
