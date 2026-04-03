from fastapi import FastAPI, Request, Depends
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
from typing import Dict, List
from contextlib import asynccontextmanager
import time
import orjson

from app.database import get_db, init_db
from app.models import TrafficResponse, BlockedRequest, ProtectionEvent
from app.config import get_settings

settings = get_settings()
start_time = time.time()
requests_processed = 0

BLOCK_SIGNATURES = {
    "sql_injection": ["SELECT", "UNION", "DROP", "INSERT", "DELETE", "UPDATE", "--", "OR '1'='1", "AND '1'='1"],
    "xss": ["<script>", "javascript:", "onerror=", "onload=", "<svg", "<iframe", "alert("],
    "path_traversal": ["../", "..\\", "%2e%2e", "/etc/passwd", "win.ini"],
    "cmd_injection": ["; ls", "| cat", "& whoami", "`id`", "$(", "; nc ", "| wget"],
    "xxe": ["<!DOCTYPE", "<!ENTITY", "SYSTEM"],
    "ssrf": ["localhost", "127.0.0.1", "169.254.169.254", "file://", "gopher://"],
}

MALICIOUS_USER_AGENTS = ["sqlmap", "nikto", "nessus", "masscan", "nmap", "dirbuster", "gobuster", "wfuzz", "hydra", "metasploit"]

def detect_attack(payload: Dict, headers: Dict) -> tuple[bool, str, str]:
    payload_str = orjson.dumps(payload).decode().lower()
    
    for attack_type, signatures in BLOCK_SIGNATURES.items():
        for sig in signatures:
            if sig.lower() in payload_str:
                return True, "nemesida_waf", f"{attack_type}:{sig}"
    
    user_agent = headers.get("User-Agent", "").lower()
    for mal_ua in MALICIOUS_USER_AGENTS:
        if mal_ua in user_agent:
            return True, "nemesida_waf", f"malicious_ua:{mal_ua}"
    
    xff = headers.get("X-Forwarded-For", "")
    if xff and ("127.0.0.1" in xff or "localhost" in xff):
        return True, "pfsense", "spoofed_ip"
    
    return False, "", ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Traffic Receiver", default_response_class=ORJSONResponse, lifespan=lifespan)

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

@app.get("/")
async def health():
    global requests_processed
    return {
        "status": "running",
        "service": "receiver",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - start_time,
        "requests_processed": requests_processed
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "receiver"}

@app.post("/receive")
async def receive_traffic(request: Request, db: AsyncSession = Depends(get_db)):
    global requests_processed
    receive_time = datetime.utcnow()
    client_ip = get_client_ip(request)
    
    try:
        body = await request.json()
    except:
        return {"error": "Invalid JSON", "status": "rejected"}
    
    batch_id = body.get("batch_id", "unknown")
    session_id = body.get("session_id", "unknown")
    requests_data = body.get("requests", [body])
    
    if not isinstance(requests_data, list):
        requests_data = [requests_data]
    
    results = []
    
    for req_data in requests_data:
        request_id = req_data.get("request_id", f"unknown-{time.time()}")
        sent_timestamp = req_data.get("timestamp", "")
        payload = req_data.get("payload", {})
        headers = req_data.get("headers", {})
        is_malicious_flag = req_data.get("is_malicious", False)
        attack_type = req_data.get("attack_type", "normal")
        
        try:
            # Parse timestamp and convert to naive UTC
            sent_time = datetime.fromisoformat(sent_timestamp.replace("Z", "+00:00"))
            if sent_time.tzinfo is not None:
                sent_time = sent_time.replace(tzinfo=None)
            response_time_ms = (receive_time - sent_time).total_seconds() * 1000
        except:
            response_time_ms = 0
        
        was_blocked, blocked_by, block_reason = detect_attack(payload, headers)
        
        response = TrafficResponse(
            request_id=request_id,
            batch_id=batch_id,
            received_at=receive_time,
            response_time_ms=response_time_ms,
            status_code=403 if was_blocked else 200,
            was_blocked=was_blocked,
            blocked_by=blocked_by if was_blocked else None,
            source_ip=client_ip,
            passed_through=not was_blocked
        )
        db.add(response)
        
        if was_blocked:
            blocked = BlockedRequest(
                request_id=request_id,
                session_id=session_id,
                blocked_by=blocked_by,
                block_reason=block_reason,
                source_ip=client_ip,
                attack_signature=attack_type
            )
            db.add(blocked)
            
            event = ProtectionEvent(
                session_id=session_id,
                event_type="block",
                source=blocked_by,
                details=f"Blocked {attack_type}: {block_reason}",
                severity="high" if is_malicious_flag else "medium",
                source_ip=client_ip,
                action_taken="blocked"
            )
            db.add(event)
        
        requests_processed += 1
        
        results.append({
            "request_id": request_id,
            "received": True,
            "response_time_ms": round(response_time_ms, 2),
            "was_blocked": was_blocked,
            "blocked_by": blocked_by if was_blocked else None,
            "status_code": 403 if was_blocked else 200
        })
    
    await db.commit()
    
    return {
        "batch_id": batch_id,
        "session_id": session_id,
        "total_requests": len(results),
        "received_count": len([r for r in results if r["received"]]),
        "blocked_count": len([r for r in results if r["was_blocked"]]),
        "timestamp": receive_time.isoformat() + "Z",
        "results": results
    }

@app.get("/stats/{session_id}")
async def get_session_stats(session_id: str, db: AsyncSession = Depends(get_db)):
    total = await db.execute(select(func.count()).where(TrafficResponse.batch_id.like(f"%{session_id}%")))
    total_count = total.scalar() or 0
    
    blocked = await db.execute(select(func.count()).where(
        TrafficResponse.batch_id.like(f"%{session_id}%"),
        TrafficResponse.was_blocked == True
    ))
    blocked_count = blocked.scalar() or 0
    
    avg_latency = await db.execute(select(func.avg(TrafficResponse.response_time_ms)).where(
        TrafficResponse.batch_id.like(f"%{session_id}%")
    ))
    avg_lat = avg_latency.scalar() or 0
    
    blocked_by = await db.execute(select(
        TrafficResponse.blocked_by, func.count()
    ).where(
        TrafficResponse.batch_id.like(f"%{session_id}%"),
        TrafficResponse.was_blocked == True
    ).group_by(TrafficResponse.blocked_by))
    
    blocked_by_stats = {row[0]: row[1] for row in blocked_by if row[0]}
    
    return {
        "session_id": session_id,
        "total_received": total_count,
        "total_blocked": blocked_count,
        "block_rate_percent": blocked_count / total_count * 100 if total_count > 0 else 0,
        "avg_response_time_ms": round(avg_lat, 2),
        "blocked_by": blocked_by_stats
    }

@app.get("/events/{session_id}")
async def get_protection_events(session_id: str, limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ProtectionEvent)
        .where(ProtectionEvent.session_id == session_id)
        .order_by(ProtectionEvent.timestamp.desc())
        .limit(limit)
    )
    events = result.scalars().all()
    
    return {
        "session_id": session_id,
        "events": [{
            "event_type": e.event_type,
            "source": e.source,
            "timestamp": e.timestamp.isoformat(),
            "details": e.details,
            "severity": e.severity,
            "action_taken": e.action_taken
        } for e in events]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.receiver:app", host=settings.receiver_host, port=settings.receiver_port, reload=True)
