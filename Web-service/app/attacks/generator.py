import random
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from app.attacks.patterns import (
    AttackCategory, get_random_pattern, generate_random_payload,
    generate_malicious_headers, get_attack_categories)


class TrafficMode(str, Enum):
    NORMAL = "normal"
    FLOOD = "flood"
    SLOWLORIS = "slowloris"
    BURST = "burst"
    GRADUAL = "gradual"
    MIXED = "mixed"

@dataclass
class TrafficConfig:
    mode: TrafficMode = TrafficMode.NORMAL
    requests_per_second: int = 100
    total_requests: int = 1000
    duration_seconds: Optional[int] = None
    burst_size: int = 50
    burst_interval_ms: int = 100
    payload_size: int = 1024
    malicious_ratio: float = 0.1
    attack_types: List[AttackCategory] = None
    slowloris_delay_ms: int = 500
    gradual_ramp_seconds: int = 60
    
    def __post_init__(self):
        if self.attack_types is None:
            self.attack_types = [AttackCategory.NORMAL]

def generate_request_id() -> str:
    return f"REQ-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

def generate_batch_id() -> str:
    return f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

def generate_normal_payload(size: int) -> Dict[str, Any]:
    return {
        "transaction_id": str(uuid.uuid4()),
        "amount": round(random.uniform(100, 50000), 2),
        "currency": "RUB",
        "sender": f"user_{random.randint(1, 1000)}",
        "receiver": f"user_{random.randint(1, 1000)}",
        "description": generate_random_payload(min(size, 200)),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def generate_malicious_payload(attack_type: AttackCategory, size: int) -> Dict[str, Any]:
    pattern, pattern_type, meta = get_random_pattern(attack_type)
    payload = generate_normal_payload(size)
    
    if attack_type == AttackCategory.SQL_INJECTION:
        payload[meta.get("field", "id")] = pattern
    elif attack_type == AttackCategory.XSS:
        payload[meta.get("field", "description")] = pattern
    elif attack_type == AttackCategory.PATH_TRAVERSAL:
        payload["file"] = pattern
    elif attack_type == AttackCategory.COMMAND_INJECTION:
        payload[meta.get("field", "cmd")] = pattern
    elif attack_type == AttackCategory.XXE:
        payload["xml_data"] = pattern
    elif attack_type == AttackCategory.SSRF:
        payload[meta.get("param", "url")] = pattern
    
    payload["_attack_type"] = pattern_type
    payload["_pattern"] = pattern[:100]
    return payload

def generate_request(config: TrafficConfig, is_malicious: bool = False) -> Dict[str, Any]:
    request_id = generate_request_id()
    
    if is_malicious and config.attack_types:
        attack_type = random.choice([a for a in config.attack_types if a != AttackCategory.NORMAL])
        if attack_type == AttackCategory.NORMAL:
            attack_type = random.choice(get_attack_categories())
        payload = generate_malicious_payload(attack_type, config.payload_size)
        headers = generate_malicious_headers()
        pattern = payload.get("_pattern", "")
        attack_name = attack_type.value
    else:
        payload = generate_normal_payload(config.payload_size)
        headers = {"User-Agent": "BankClient/1.0", "Content-Type": "application/json"}
        pattern = None
        attack_name = "normal"
    
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "payload_size": len(str(payload)),
        "is_malicious": is_malicious,
        "attack_type": attack_name,
        "malicious_pattern": pattern,
        "headers": headers
    }

async def generate_normal_traffic(config: TrafficConfig, batch_id: str) -> AsyncGenerator[Dict, None]:
    interval = 1.0 / config.requests_per_second
    count = 0
    while count < config.total_requests:
        is_malicious = random.random() < config.malicious_ratio
        yield generate_request(config, is_malicious)
        count += 1
        await asyncio.sleep(interval)

async def generate_flood_traffic(config: TrafficConfig, batch_id: str) -> AsyncGenerator[Dict, None]:
    for _ in range(config.total_requests):
        is_malicious = random.random() < config.malicious_ratio
        yield generate_request(config, is_malicious)

async def generate_burst_traffic(config: TrafficConfig, batch_id: str) -> AsyncGenerator[Dict, None]:
    count = 0
    while count < config.total_requests:
        for _ in range(min(config.burst_size, config.total_requests - count)):
            is_malicious = random.random() < config.malicious_ratio
            yield generate_request(config, is_malicious)
            count += 1
        await asyncio.sleep(config.burst_interval_ms / 1000)

async def generate_slowloris_traffic(config: TrafficConfig, batch_id: str) -> AsyncGenerator[Dict, None]:
    for _ in range(config.total_requests):
        request = generate_request(config, random.random() < config.malicious_ratio)
        request["slowloris"] = True
        request["partial_headers"] = True
        yield request
        await asyncio.sleep(config.slowloris_delay_ms / 1000)

async def generate_gradual_traffic(config: TrafficConfig, batch_id: str) -> AsyncGenerator[Dict, None]:
    count = 0
    elapsed = 0
    ramp_duration = config.gradual_ramp_seconds
    max_rps = config.requests_per_second
    
    while count < config.total_requests:
        current_rps = min(max_rps, max_rps * (elapsed / ramp_duration)) if elapsed < ramp_duration else max_rps
        current_rps = max(1, current_rps)
        interval = 1.0 / current_rps
        is_malicious = random.random() < config.malicious_ratio
        yield generate_request(config, is_malicious)
        count += 1
        await asyncio.sleep(interval)
        elapsed += interval

async def generate_mixed_traffic(config: TrafficConfig, batch_id: str) -> AsyncGenerator[Dict, None]:
    modes = [TrafficMode.NORMAL, TrafficMode.BURST, TrafficMode.FLOOD]
    requests_per_mode = config.total_requests // len(modes)
    
    for mode in modes:
        sub_config = TrafficConfig(
            mode=mode,
            requests_per_second=config.requests_per_second,
            total_requests=requests_per_mode,
            burst_size=config.burst_size,
            burst_interval_ms=config.burst_interval_ms,
            payload_size=config.payload_size,
            malicious_ratio=config.malicious_ratio,
            attack_types=config.attack_types
        )
        
        if mode == TrafficMode.NORMAL:
            async for req in generate_normal_traffic(sub_config, batch_id):
                yield req
        elif mode == TrafficMode.BURST:
            async for req in generate_burst_traffic(sub_config, batch_id):
                yield req
        elif mode == TrafficMode.FLOOD:
            async for req in generate_flood_traffic(sub_config, batch_id):
                yield req

def get_traffic_generator(config: TrafficConfig, batch_id: str):
    generators = {
        TrafficMode.NORMAL: generate_normal_traffic,
        TrafficMode.FLOOD: generate_flood_traffic,
        TrafficMode.BURST: generate_burst_traffic,
        TrafficMode.SLOWLORIS: generate_slowloris_traffic,
        TrafficMode.GRADUAL: generate_gradual_traffic,
        TrafficMode.MIXED: generate_mixed_traffic,
    }
    return generators.get(config.mode, generate_normal_traffic)(config, batch_id)
