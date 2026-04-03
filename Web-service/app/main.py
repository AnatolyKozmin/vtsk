import asyncio
import argparse
import time
import httpx
from datetime import datetime, timezone

from app.config import get_settings
from app.attacks.generator import TrafficConfig, TrafficMode, get_traffic_generator, generate_batch_id
from app.attacks.patterns import AttackCategory
from app.services.metrics import MetricsCollector

settings = get_settings()

async def run_test(
    mode: str = "normal",
    total_requests: int = 1000,
    requests_per_second: int = 100,
    malicious_ratio: float = 0.1,
    attack_types: str = "sql_injection,xss",
    payload_size: int = 1024,
    burst_size: int = 50
):
    print("=" * 70)
    print("PROTECTION TEST - TRAFFIC GENERATOR")
    print("=" * 70)
    print(f"Target: {settings.receiver_url}")
    print(f"Mode: {mode}")
    print(f"Total requests: {total_requests}")
    print(f"RPS: {requests_per_second}")
    print(f"Malicious ratio: {malicious_ratio * 100}%")
    print(f"Attack types: {attack_types}")
    print("=" * 70)
    
    try:
        traffic_mode = TrafficMode(mode)
    except ValueError:
        traffic_mode = TrafficMode.NORMAL
    
    attack_list = []
    for at in attack_types.split(","):
        try:
            attack_list.append(AttackCategory(at.strip()))
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
        attack_types=attack_list
    )
    
    session_id = f"CLI-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    batch_id = generate_batch_id()
    metrics = MetricsCollector(session_id)
    metrics.start()
    
    print(f"\nSession: {session_id}")
    print("Starting test...\n")
    
    start_time = time.time()
    sent_count = 0
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        if traffic_mode == TrafficMode.FLOOD:
            semaphore = asyncio.Semaphore(100)
            tasks = []
            
            async def send_request(req_data):
                async with semaphore:
                    request_id = req_data["request_id"]
                    metrics.record_sent(request_id, req_data.get("is_malicious", False), req_data.get("attack_type", "normal"))
                    send_start = time.time()
                    try:
                        payload = {"batch_id": batch_id, "session_id": session_id, "requests": [req_data]}
                        resp = await client.post(f"{settings.receiver_url}/receive", json=payload)
                        response_time_ms = (time.time() - send_start) * 1000
                        if resp.status_code < 400:
                            result = resp.json()
                            results = result.get("results", [])
                            if results:
                                r = results[0]
                                metrics.record_received(request_id, response_time_ms, r.get("status_code", 200),
                                                       r.get("was_blocked", False), r.get("blocked_by"))
                        else:
                            metrics.record_received(request_id, response_time_ms, resp.status_code, error=f"HTTP {resp.status_code}")
                    except Exception as e:
                        metrics.record_received(request_id, (time.time() - send_start) * 1000, 0, error=str(e))
            
            async for request_data in get_traffic_generator(config, batch_id):
                tasks.append(send_request(request_data))
                sent_count += 1
                if len(tasks) >= 100:
                    await asyncio.gather(*tasks)
                    tasks = []
                    elapsed = time.time() - start_time
                    print(f"\rSent: {sent_count}/{total_requests} | Rate: {sent_count/elapsed:.1f} req/s", end="")
            if tasks:
                await asyncio.gather(*tasks)
        else:
            async for request_data in get_traffic_generator(config, batch_id):
                request_id = request_data["request_id"]
                metrics.record_sent(request_id, request_data.get("is_malicious", False), request_data.get("attack_type", "normal"))
                send_start = time.time()
                try:
                    payload = {"batch_id": batch_id, "session_id": session_id, "requests": [request_data]}
                    resp = await client.post(f"{settings.receiver_url}/receive", json=payload)
                    response_time_ms = (time.time() - send_start) * 1000
                    if resp.status_code < 400:
                        result = resp.json()
                        results = result.get("results", [])
                        if results:
                            r = results[0]
                            metrics.record_received(request_id, response_time_ms, r.get("status_code", 200),
                                                   r.get("was_blocked", False), r.get("blocked_by"))
                except Exception as e:
                    metrics.record_received(request_id, (time.time() - send_start) * 1000, 0, error=str(e))
                
                sent_count += 1
                if sent_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"\rSent: {sent_count}/{total_requests} | Rate: {sent_count/elapsed:.1f} req/s", end="")
    
    total_time = time.time() - start_time
    summary = metrics.get_summary()
    protection = metrics.get_protection_effectiveness()
    
    print("\n\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"\nDuration: {total_time:.2f}s")
    print(f"Throughput: {summary['throughput_rps']:.1f} req/s")
    print(f"\n--- Traffic Stats ---")
    print(f"Total sent: {summary['total_sent']}")
    print(f"Total received: {summary['total_received']}")
    print(f"Total blocked: {summary['total_blocked']}")
    print(f"Block rate: {summary['block_rate']:.1f}%")
    print(f"\n--- Latency (ms) ---")
    lat = summary['latency']
    print(f"Avg: {lat['avg_ms']:.2f} | Min: {lat['min_ms']:.2f} | Max: {lat['max_ms']:.2f}")
    print(f"P50: {lat['p50_ms']:.2f} | P95: {lat['p95_ms']:.2f} | P99: {lat['p99_ms']:.2f}")
    print(f"\n--- Protection Effectiveness ---")
    print(f"Malicious sent: {protection['malicious_sent']}")
    print(f"Malicious blocked: {protection['malicious_blocked']}")
    print(f"Malicious passed: {protection['malicious_passed']}")
    print(f"Detection rate: {protection['detection_rate_percent']:.1f}%")
    print(f"False positive rate: {protection['false_positive_rate_percent']:.1f}%")
    
    if summary['blocked_by']:
        print(f"\n--- Blocked By ---")
        for source, count in summary['blocked_by'].items():
            print(f"  {source}: {count}")
    
    if summary['attack_stats']:
        print(f"\n--- Attack Type Stats ---")
        for attack_type, stats in summary['attack_stats'].items():
            if stats['sent'] > 0:
                block_rate = stats['blocked'] / stats['sent'] * 100
                print(f"  {attack_type}: sent={stats['sent']}, blocked={stats['blocked']} ({block_rate:.1f}%)")
    
    print("\n" + "=" * 70)
    if protection['detection_rate_percent'] < 80:
        print("WARNING: Detection rate below 80%")
    if protection['false_positive_rate_percent'] > 5:
        print("WARNING: False positive rate above 5%")
    if protection['malicious_passed'] > 0:
        print(f"WARNING: {protection['malicious_passed']} malicious requests passed through")
    print("\nDone!")

def main():
    parser = argparse.ArgumentParser(description='Protection Test - Traffic Generator')
    parser.add_argument('--mode', choices=['normal', 'flood', 'burst', 'slowloris', 'gradual', 'mixed'], default='normal')
    parser.add_argument('--requests', type=int, default=1000)
    parser.add_argument('--rps', type=int, default=100)
    parser.add_argument('--malicious', type=float, default=0.1)
    parser.add_argument('--attacks', type=str, default='sql_injection,xss')
    parser.add_argument('--payload-size', type=int, default=1024)
    parser.add_argument('--burst-size', type=int, default=50)
    args = parser.parse_args()
    
    asyncio.run(run_test(
        mode=args.mode,
        total_requests=args.requests,
        requests_per_second=args.rps,
        malicious_ratio=args.malicious,
        attack_types=args.attacks,
        payload_size=args.payload_size,
        burst_size=args.burst_size
    ))

if __name__ == "__main__":
    main()
