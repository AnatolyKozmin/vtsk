import random
import asyncio
import struct
import socket
from typing import Dict, List, Generator, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

@dataclass
class BotnetNode:
    ip: str
    geo_location: str
    bot_type: str
    bandwidth_mbps: float
    latency_ms: float

class RealisticDDoSGenerator:
    def __init__(self):
        self.botnet_nodes: List[BotnetNode] = []
        self.attack_distributions = {
            "syn_flood": {"packet_size": (40, 60), "rate_variance": 0.3},
            "udp_flood": {"packet_size": (512, 1400), "rate_variance": 0.5},
            "http_flood": {"packet_size": (200, 2000), "rate_variance": 0.4},
            "dns_amplification": {"amplification_factor": (28, 54), "rate_variance": 0.2},
            "ntp_amplification": {"amplification_factor": (200, 556), "rate_variance": 0.2},
            "slowloris": {"connection_duration": (30, 300), "rate_variance": 0.1}
        }
    
    def generate_botnet(self, size: int, geo_distribution: Dict[str, float] = None) -> List[BotnetNode]:
        if geo_distribution is None:
            geo_distribution = {
                "US": 0.25, "CN": 0.20, "RU": 0.15, "BR": 0.10,
                "IN": 0.10, "DE": 0.05, "FR": 0.05, "OTHER": 0.10
            }
        
        self.botnet_nodes = []
        
        for _ in range(size):
            geo = random.choices(list(geo_distribution.keys()), 
                               weights=list(geo_distribution.values()))[0]
            
            ip = self._generate_ip_for_region(geo)
            
            bandwidth = np.random.lognormal(mean=2, sigma=1)
            bandwidth = min(max(bandwidth, 0.5), 100)
            
            latency = self._get_latency_for_region(geo)
            
            self.botnet_nodes.append(BotnetNode(
                ip=ip,
                geo_location=geo,
                bot_type=random.choice(["iot", "server", "desktop", "mobile"]),
                bandwidth_mbps=bandwidth,
                latency_ms=latency
            ))
        
        return self.botnet_nodes
    
    def _generate_ip_for_region(self, region: str) -> str:
        region_ranges = {
            "US": [(3, 126), (128, 191)],
            "CN": [(1, 126), (202, 223)],
            "RU": [(5, 95), (176, 191)],
            "BR": [(177, 191), (200, 201)],
            "IN": [(14, 14), (27, 27), (49, 49)],
            "DE": [(5, 5), (31, 31), (46, 46)],
            "FR": [(2, 2), (5, 5), (31, 31)],
            "OTHER": [(1, 223)]
        }
        
        ranges = region_ranges.get(region, region_ranges["OTHER"])
        first_octet_range = random.choice(ranges)
        
        return f"{random.randint(*first_octet_range)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    
    def _get_latency_for_region(self, region: str) -> float:
        base_latencies = {
            "US": 50, "CN": 200, "RU": 150, "BR": 180,
            "IN": 250, "DE": 30, "FR": 25, "OTHER": 100
        }
        base = base_latencies.get(region, 100)
        return base + np.random.exponential(scale=base * 0.3)
    
    def generate_syn_flood_packet(self, source_ip: str = None) -> Dict:
        if source_ip is None:
            source_ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        
        return {
            "type": "syn_flood",
            "source_ip": source_ip,
            "source_port": random.randint(1024, 65535),
            "dest_port": random.choice([80, 443, 8080, 8443]),
            "tcp_flags": {"SYN": True, "ACK": False, "FIN": False, "RST": False},
            "window_size": random.choice([1024, 2048, 4096, 8192, 16384, 29200, 65535]),
            "packet_size": random.randint(40, 60),
            "ttl": random.randint(32, 128),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def generate_udp_flood_packet(self, source_ip: str = None) -> Dict:
        if source_ip is None:
            source_ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        
        packet_size = random.randint(512, 1400)
        
        return {
            "type": "udp_flood",
            "source_ip": source_ip,
            "source_port": random.randint(1024, 65535),
            "dest_port": random.choice([53, 123, 161, 1900, 11211]),
            "packet_size": packet_size,
            "payload": "X" * packet_size,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def generate_dns_amplification_request(self) -> Dict:
        dns_servers = [
            "8.8.8.8", "8.8.4.4", "1.1.1.1", "9.9.9.9",
            "208.67.222.222", "208.67.220.220"
        ]
        
        query_types = ["ANY", "TXT", "DNSKEY", "RRSIG"]
        domains = ["google.com", "facebook.com", "amazon.com", "cloudflare.com"]
        
        amplification = random.randint(28, 54)
        
        return {
            "type": "dns_amplification",
            "dns_server": random.choice(dns_servers),
            "query_type": random.choice(query_types),
            "query_domain": random.choice(domains),
            "request_size": 60,
            "response_size": 60 * amplification,
            "amplification_factor": amplification,
            "spoofed_source": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def generate_ntp_amplification_request(self) -> Dict:
        amplification = random.randint(200, 556)
        
        return {
            "type": "ntp_amplification",
            "command": "monlist",
            "request_size": 8,
            "response_size": 8 * amplification,
            "amplification_factor": amplification,
            "spoofed_source": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def generate_slowloris_connection(self) -> Dict:
        headers = [
            "X-a: b",
            "X-custom: value",
            "Accept-Language: en-US",
            "Cache-Control: no-cache"
        ]
        
        return {
            "type": "slowloris",
            "partial_request": f"GET / HTTP/1.1\r\nHost: target.com\r\n{random.choice(headers)}\r\n",
            "connection_state": "partial",
            "bytes_sent": random.randint(50, 200),
            "keep_alive_interval_ms": random.randint(5000, 15000),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def generate_realistic_attack_stream(self, attack_type: str, 
                                               duration_seconds: int,
                                               intensity_pps: int) -> Generator:
        if not self.botnet_nodes:
            self.generate_botnet(1000)
        
        generators = {
            "syn_flood": self.generate_syn_flood_packet,
            "udp_flood": self.generate_udp_flood_packet,
            "dns_amplification": self.generate_dns_amplification_request,
            "ntp_amplification": self.generate_ntp_amplification_request,
            "slowloris": self.generate_slowloris_connection
        }
        
        generator_func = generators.get(attack_type, self.generate_syn_flood_packet)
        config = self.attack_distributions.get(attack_type, {"rate_variance": 0.3})
        
        start_time = datetime.now()
        packets_sent = 0
        
        while (datetime.now() - start_time).total_seconds() < duration_seconds:
            current_rate = intensity_pps * (1 + random.uniform(-config["rate_variance"], config["rate_variance"]))
            
            batch_size = max(1, int(current_rate / 100))
            
            for _ in range(batch_size):
                bot = random.choice(self.botnet_nodes)
                
                if attack_type in ["syn_flood", "udp_flood"]:
                    packet = generator_func(bot.ip)
                else:
                    packet = generator_func()
                
                packet["bot_info"] = {
                    "geo": bot.geo_location,
                    "type": bot.bot_type,
                    "latency": bot.latency_ms
                }
                
                yield packet
                packets_sent += 1
            
            await asyncio.sleep(0.01)
        
        yield {"type": "stream_end", "total_packets": packets_sent}
    
    def get_attack_statistics(self, packets: List[Dict]) -> Dict:
        if not packets:
            return {}
        
        source_ips = set()
        geo_distribution = {}
        packet_sizes = []
        
        for p in packets:
            if "source_ip" in p:
                source_ips.add(p["source_ip"])
            
            if "bot_info" in p:
                geo = p["bot_info"].get("geo", "unknown")
                geo_distribution[geo] = geo_distribution.get(geo, 0) + 1
            
            if "packet_size" in p:
                packet_sizes.append(p["packet_size"])
        
        return {
            "total_packets": len(packets),
            "unique_sources": len(source_ips),
            "geo_distribution": geo_distribution,
            "packet_size_stats": {
                "mean": np.mean(packet_sizes) if packet_sizes else 0,
                "std": np.std(packet_sizes) if packet_sizes else 0,
                "min": min(packet_sizes) if packet_sizes else 0,
                "max": max(packet_sizes) if packet_sizes else 0
            },
            "source_diversity_ratio": len(source_ips) / len(packets) if packets else 0
        }
