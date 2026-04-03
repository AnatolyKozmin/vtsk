import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ZabbixHost:
    host_id: str
    name: str
    ip: str
    status: str


@dataclass
class ZabbixMetric:
    item_id: str
    name: str
    key: str
    value: float
    timestamp: datetime
    host: str


class ZabbixClient:
    def __init__(self, url: str, user: str, password: str):
        self.url = url.rstrip('/') + '/api_jsonrpc.php'
        self.user = user
        self.password = password
        self.auth_token: Optional[str] = None
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def _request(self, method: str, params: Dict = None) -> Dict:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }
        
        if self.auth_token and method != "user.login":
            payload["auth"] = self.auth_token
        
        response = await self.client.post(self.url, json=payload)
        result = response.json()
        
        if "error" in result:
            raise Exception(f"Zabbix API error: {result['error']}")
        
        return result.get("result", {})
    
    async def login(self) -> bool:
        try:
            result = await self._request("user.login", {
                "user": self.user,
                "password": self.password
            })
            self.auth_token = result
            return True
        except:
            return False
    
    async def logout(self):
        if self.auth_token:
            await self._request("user.logout")
            self.auth_token = None
    
    async def get_hosts(self, group_name: Optional[str] = None) -> List[ZabbixHost]:
        params = {
            "output": ["hostid", "host", "name", "status"],
            "selectInterfaces": ["ip"]
        }
        
        if group_name:
            groups = await self._request("hostgroup.get", {"filter": {"name": group_name}})
            if groups:
                params["groupids"] = [g["groupid"] for g in groups]
        
        result = await self._request("host.get", params)
        
        hosts = []
        for h in result:
            ip = h.get("interfaces", [{}])[0].get("ip", "unknown")
            hosts.append(ZabbixHost(
                host_id=h["hostid"],
                name=h["name"],
                ip=ip,
                status="enabled" if h["status"] == "0" else "disabled"
            ))
        
        return hosts
    
    async def get_items(self, host_id: str, search_key: Optional[str] = None) -> List[Dict]:
        params = {
            "output": ["itemid", "name", "key_", "lastvalue", "lastclock"],
            "hostids": host_id
        }
        
        if search_key:
            params["search"] = {"key_": search_key}
        
        return await self._request("item.get", params)
    
    async def get_history(self, item_id: str, time_from: datetime, 
                         time_till: datetime, history_type: int = 0) -> List[Dict]:
        params = {
            "output": "extend",
            "itemids": item_id,
            "time_from": int(time_from.timestamp()),
            "time_till": int(time_till.timestamp()),
            "history": history_type,
            "sortfield": "clock",
            "sortorder": "ASC"
        }
        
        return await self._request("history.get", params)
    
    async def get_pfsense_metrics(self, host_id: str) -> Dict[str, float]:
        metrics = {}
        
        key_mappings = {
            "system.cpu.util": "cpu_utilization",
            "vm.memory.util": "memory_utilization",
            "net.if.in": "network_in_bytes",
            "net.if.out": "network_out_bytes",
            "pfsense.states.count": "firewall_states",
            "pfsense.rules.blocked": "blocked_packets",
            "pfsense.rules.passed": "passed_packets"
        }
        
        items = await self.get_items(host_id)
        
        for item in items:
            key = item.get("key_", "")
            for zabbix_key, metric_name in key_mappings.items():
                if zabbix_key in key:
                    try:
                        metrics[metric_name] = float(item.get("lastvalue", 0))
                    except:
                        metrics[metric_name] = 0
        
        return metrics
    
    async def get_waf_metrics(self, host_id: str) -> Dict[str, float]:
        metrics = {}
        
        key_mappings = {
            "nemesida.requests.total": "total_requests",
            "nemesida.requests.blocked": "blocked_requests",
            "nemesida.attacks.sqli": "sql_injection_attacks",
            "nemesida.attacks.xss": "xss_attacks",
            "nemesida.attacks.rce": "rce_attacks",
            "nemesida.latency.avg": "avg_latency",
            "nemesida.cpu": "waf_cpu_usage"
        }
        
        items = await self.get_items(host_id)
        
        for item in items:
            key = item.get("key_", "")
            for zabbix_key, metric_name in key_mappings.items():
                if zabbix_key in key:
                    try:
                        metrics[metric_name] = float(item.get("lastvalue", 0))
                    except:
                        metrics[metric_name] = 0
        
        return metrics
    
    async def get_network_throughput(self, host_id: str, interface: str = "em0",
                                    time_range_minutes: int = 5) -> Dict:
        time_till = datetime.now()
        time_from = time_till - timedelta(minutes=time_range_minutes)
        
        items = await self.get_items(host_id, f"net.if.in[{interface}]")
        
        if not items:
            return {"in_bps": 0, "out_bps": 0, "total_bps": 0}
        
        in_item = items[0]
        history = await self.get_history(in_item["itemid"], time_from, time_till, 3)
        
        if len(history) < 2:
            return {"in_bps": 0, "out_bps": 0, "total_bps": 0}
        
        values = [float(h["value"]) for h in history]
        avg_in = sum(values) / len(values)
        
        return {
            "in_bps": avg_in,
            "out_bps": avg_in * 0.8,
            "total_bps": avg_in * 1.8,
            "samples": len(values)
        }
    
    async def close(self):
        await self.client.aclose()

class ZabbixMetricsCollector:
    def __init__(self, zabbix_url: str, user: str, password: str):
        self.client = ZabbixClient(zabbix_url, user, password)
        self.pfsense_host_id: Optional[str] = None
        self.waf_host_id: Optional[str] = None
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_ttl = 5
    
    async def initialize(self, pfsense_host: str, waf_host: str):
        await self.client.login()
        
        hosts = await self.client.get_hosts()
        for host in hosts:
            if pfsense_host.lower() in host.name.lower():
                self.pfsense_host_id = host.host_id
            if waf_host.lower() in host.name.lower():
                self.waf_host_id = host.host_id
    
    async def collect_all_metrics(self) -> Dict:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pfsense": {},
            "waf": {},
            "network": {}
        }
        
        if self.pfsense_host_id:
            metrics["pfsense"] = await self.client.get_pfsense_metrics(self.pfsense_host_id)
            metrics["network"] = await self.client.get_network_throughput(self.pfsense_host_id)
        
        if self.waf_host_id:
            metrics["waf"] = await self.client.get_waf_metrics(self.waf_host_id)
        
        self.metrics_cache = metrics
        return metrics
    
    async def get_protection_load_correlation(self) -> Dict:
        metrics = await self.collect_all_metrics()
        
        pfsense = metrics.get("pfsense", {})
        waf = metrics.get("waf", {})
        network = metrics.get("network", {})
        
        total_blocked = pfsense.get("blocked_packets", 0) + waf.get("blocked_requests", 0)
        total_passed = pfsense.get("passed_packets", 0) + waf.get("total_requests", 0) - waf.get("blocked_requests", 0)
        
        return {
            "infrastructure_load": {
                "pfsense_cpu": pfsense.get("cpu_utilization", 0),
                "pfsense_memory": pfsense.get("memory_utilization", 0),
                "waf_cpu": waf.get("waf_cpu_usage", 0),
                "network_throughput_mbps": network.get("total_bps", 0) / 1_000_000
            },
            "protection_stats": {
                "total_blocked": total_blocked,
                "total_passed": total_passed,
                "block_rate": total_blocked / (total_blocked + total_passed) * 100 if (total_blocked + total_passed) > 0 else 0
            },
            "attack_breakdown": {
                "sql_injection": waf.get("sql_injection_attacks", 0),
                "xss": waf.get("xss_attacks", 0),
                "rce": waf.get("rce_attacks", 0)
            }
        }
    
    async def close(self):
        await self.client.logout()
        await self.client.close()
