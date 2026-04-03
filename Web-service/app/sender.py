from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from typing import Dict, Any
from contextlib import asynccontextmanager
import asyncio
import time
import uuid
import httpx
import orjson

from app.database import get_db, init_db
from app.models import TestSession
from app.config import get_settings
from app.attacks.generator import TrafficConfig, TrafficMode, get_traffic_generator, generate_batch_id
from app.attacks.patterns import AttackCategory
from app.services.metrics import MetricsCollector

settings = get_settings()
start_time = time.time()
active_sessions: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Traffic Sender", default_response_class=ORJSONResponse, lifespan=lifespan)

async def send_request(client: httpx.AsyncClient, request_data: Dict, session_id: str, metrics: MetricsCollector) -> Dict:
    request_id = request_data["request_id"]
    metrics.record_sent(request_id, request_data.get("is_malicious", False), request_data.get("attack_type", "normal"))
    send_start = time.time()
    
    try:
        payload = {"batch_id": session_id, "session_id": session_id, "requests": [request_data]}
        resp = await client.post(
            f"{settings.receiver_url}/receive",
            content=orjson.dumps(payload),
            headers=request_data.get("headers", {"Content-Type": "application/json"}),
            timeout=settings.request_timeout
        )
        response_time_ms = (time.time() - send_start) * 1000
        
        if resp.status_code < 400:
            result = resp.json()
            results = result.get("results", [])
            if results:
                r = results[0]
                metrics.record_received(request_id, response_time_ms, r.get("status_code", 200),
                                       r.get("was_blocked", False), r.get("blocked_by"))
                return r
        
        metrics.record_received(request_id, response_time_ms, resp.status_code, error=f"HTTP {resp.status_code}")
        return {"request_id": request_id, "received": False, "error": f"HTTP {resp.status_code}"}
    except httpx.TimeoutException:
        metrics.record_received(request_id, (time.time() - send_start) * 1000, 0, error="Timeout")
        return {"request_id": request_id, "received": False, "error": "Timeout"}
    except Exception as e:
        metrics.record_received(request_id, (time.time() - send_start) * 1000, 0, error=str(e))
        return {"request_id": request_id, "received": False, "error": str(e)}

async def run_test_session(session_id: str, config: TrafficConfig, db: AsyncSession):
    metrics = MetricsCollector(session_id)
    metrics.start()
    
    active_sessions[session_id] = {
        "config": config,
        "metrics": metrics,
        "status": "running",
        "started_at": datetime.now(timezone.utc)
    }
    
    session = TestSession(
        session_id=session_id,
        name=f"Test {config.mode.value}",
        attack_type=config.mode.value,
        total_requests=config.total_requests,
        status="running"
    )
    db.add(session)
    await db.commit()
    
    batch_id = generate_batch_id()
    sent_count = 0
    
    async with httpx.AsyncClient() as client:
        if config.mode == TrafficMode.FLOOD:
            semaphore = asyncio.Semaphore(settings.max_workers)
            
            async def send_with_semaphore(req_data):
                async with semaphore:
                    return await send_request(client, req_data, session_id, metrics)
            
            tasks = []
            async for request_data in get_traffic_generator(config, batch_id):
                tasks.append(send_with_semaphore(request_data))
                sent_count += 1
                if len(tasks) >= 100:
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)
        else:
            async for request_data in get_traffic_generator(config, batch_id):
                await send_request(client, request_data, session_id, metrics)
                sent_count += 1
                if sent_count % 100 == 0:
                    active_sessions[session_id]["sent_count"] = sent_count
    
    summary = metrics.get_summary()
    
    result = await db.execute(select(TestSession).where(TestSession.session_id == session_id))
    session = result.scalar_one_or_none()
    if session:
        session.ended_at = datetime.now(timezone.utc)
        session.requests_sent = summary["total_sent"]
        session.requests_received = summary["total_received"]
        session.requests_blocked = summary["total_blocked"]
        session.avg_response_time = summary["latency"]["avg_ms"]
        session.min_response_time = summary["latency"]["min_ms"]
        session.max_response_time = summary["latency"]["max_ms"]
        session.throughput_rps = summary["throughput_rps"]
        session.status = "completed"
        await db.commit()
    
    active_sessions[session_id]["status"] = "completed"
    active_sessions[session_id]["summary"] = summary

@app.get("/")
async def health():
    return {
        "status": "running",
        "service": "sender",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "uptime_seconds": time.time() - start_time,
        "active_sessions": len([s for s in active_sessions.values() if s["status"] == "running"])
    }

@app.post("/start")
async def start_test(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    mode: str = "normal",
    total_requests: int = 1000,
    requests_per_second: int = 100,
    malicious_ratio: float = 0.1,
    payload_size: int = 1024,
    burst_size: int = 50,
    burst_interval_ms: int = 100,
    attack_types: str = "sql_injection,xss"
):
    session_id = f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    
    try:
        traffic_mode = TrafficMode(mode)
    except ValueError:
        traffic_mode = TrafficMode.NORMAL
    
    attack_list = []
    for at in attack_types.split(","):
        at = at.strip()
        try:
            attack_list.append(AttackCategory(at))
        except ValueError:
            pass
    
    if not attack_list:
        attack_list = [AttackCategory.SQL_INJECTION, AttackCategory.XSS]
    
    config = TrafficConfig(
        mode=traffic_mode,
        total_requests=total_requests,
        requests_per_second=requests_per_second,
        malicious_ratio=malicious_ratio,
        payload_size=payload_size,
        burst_size=burst_size,
        burst_interval_ms=burst_interval_ms,
        attack_types=attack_list
    )
    
    background_tasks.add_task(run_test_session, session_id, config, db)
    
    return {
        "session_id": session_id,
        "status": "started",
        "config": {
            "mode": mode,
            "total_requests": total_requests,
            "requests_per_second": requests_per_second,
            "malicious_ratio": malicious_ratio,
            "attack_types": [a.value for a in attack_list]
        },
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    }

@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    metrics = session.get("metrics")
    
    if metrics:
        summary = metrics.get_summary()
        protection = metrics.get_protection_effectiveness()
        return {
            "session_id": session_id,
            "status": session["status"],
            "started_at": session["started_at"].isoformat(),
            "summary": summary,
            "protection_effectiveness": protection
        }
    
    return {"session_id": session_id, "status": session["status"], "started_at": session["started_at"].isoformat()}

@app.get("/timeline/{session_id}")
async def get_session_timeline(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    metrics = active_sessions[session_id].get("metrics")
    if not metrics:
        return {"session_id": session_id, "timeline": []}
    
    return {"session_id": session_id, "timeline": metrics.get_timeline()}

@app.post("/stop/{session_id}")
async def stop_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    active_sessions[session_id]["status"] = "stopped"
    return {"session_id": session_id, "status": "stopped"}

@app.get("/sessions")
async def list_sessions():
    return {
        "sessions": [{
            "session_id": sid,
            "status": s["status"],
            "started_at": s["started_at"].isoformat()
        } for sid, s in active_sessions.items()]
    }

@app.get("/report/{session_id}")
async def get_report(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    metrics = session.get("metrics")
    
    if not metrics:
        raise HTTPException(status_code=400, detail="No metrics available")
    
    summary = metrics.get_summary()
    protection = metrics.get_protection_effectiveness()
    timeline = metrics.get_timeline()
    
    recommendations = []
    if protection["detection_rate_percent"] < 80:
        recommendations.append("Detection rate is below 80%. Consider tuning WAF rules.")
    if protection["false_positive_rate_percent"] > 5:
        recommendations.append("False positive rate is high. Review blocking rules.")
    if summary["latency"]["p99_ms"] > 1000:
        recommendations.append("P99 latency is high. Check network and server performance.")
    if protection["malicious_passed"] > 0:
        recommendations.append(f"{protection['malicious_passed']} malicious requests passed through. Update signatures.")
    
    return {
        "session_id": session_id,
        "name": f"Test {session['config'].mode.value}",
        "attack_type": session["config"].mode.value,
        "duration_seconds": summary["duration_seconds"],
        "summary": summary,
        "protection": protection,
        "timeline": timeline,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.sender:app", host=settings.sender_host, port=settings.sender_port, reload=True)
