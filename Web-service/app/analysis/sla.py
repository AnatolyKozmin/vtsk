import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class SLAStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"

@dataclass
class SLOTarget:
    name: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    higher_is_better: bool = False

@dataclass
class SLAMetric:
    name: str
    current_value: float
    target_value: float
    compliance_percent: float
    status: SLAStatus
    violation_duration_seconds: float
    cost_impact: float

class SLAAnalyzer:
    def __init__(self):
        self.slo_targets = {
            "availability": SLOTarget("Availability", 99.9, 99.5, 99.0, "%", True),
            "latency_p50": SLOTarget("P50 Latency", 50, 100, 200, "ms", False),
            "latency_p95": SLOTarget("P95 Latency", 200, 500, 1000, "ms", False),
            "latency_p99": SLOTarget("P99 Latency", 500, 1000, 2000, "ms", False),
            "throughput": SLOTarget("Throughput", 1000, 500, 100, "req/s", True),
            "error_rate": SLOTarget("Error Rate", 0.1, 1.0, 5.0, "%", False),
            "packet_loss": SLOTarget("Packet Loss", 0.01, 0.1, 1.0, "%", False)
        }
        self.violation_history: List[Dict] = []
        self.cost_per_violation_minute = 100.0
    
    def set_slo_target(self, name: str, target: SLOTarget):
        self.slo_targets[name] = target
    
    def _calculate_status(self, value: float, target: SLOTarget) -> SLAStatus:
        if target.higher_is_better:
            if value >= target.target_value:
                return SLAStatus.COMPLIANT
            elif value >= target.warning_threshold:
                return SLAStatus.WARNING
            elif value >= target.critical_threshold:
                return SLAStatus.VIOLATION
            else:
                return SLAStatus.CRITICAL
        else:
            if value <= target.target_value:
                return SLAStatus.COMPLIANT
            elif value <= target.warning_threshold:
                return SLAStatus.WARNING
            elif value <= target.critical_threshold:
                return SLAStatus.VIOLATION
            else:
                return SLAStatus.CRITICAL
    
    def _calculate_compliance(self, value: float, target: SLOTarget) -> float:
        if target.higher_is_better:
            if value >= target.target_value:
                return 100.0
            return (value / target.target_value) * 100
        else:
            if value <= target.target_value:
                return 100.0
            if value >= target.critical_threshold:
                return 0.0
            return ((target.critical_threshold - value) / (target.critical_threshold - target.target_value)) * 100
    
    def analyze_availability(self, total_requests: int, successful_requests: int, 
                            time_window_seconds: float) -> SLAMetric:
        availability = (successful_requests / total_requests * 100) if total_requests > 0 else 100.0
        target = self.slo_targets["availability"]
        status = self._calculate_status(availability, target)
        compliance = self._calculate_compliance(availability, target)
        
        violation_duration = 0.0
        if status in [SLAStatus.VIOLATION, SLAStatus.CRITICAL]:
            violation_duration = time_window_seconds
            self.violation_history.append({
                "metric": "availability",
                "value": availability,
                "timestamp": datetime.now(),
                "duration": violation_duration
            })
        
        cost = (violation_duration / 60) * self.cost_per_violation_minute if violation_duration > 0 else 0
        
        return SLAMetric(
            name="Availability",
            current_value=availability,
            target_value=target.target_value,
            compliance_percent=compliance,
            status=status,
            violation_duration_seconds=violation_duration,
            cost_impact=cost
        )
    
    def analyze_latency(self, latencies: List[float]) -> Dict[str, SLAMetric]:
        if not latencies:
            return {}
        
        results = {}
        percentiles = {
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99)
        }
        
        for metric_name, value in percentiles.items():
            target = self.slo_targets[metric_name]
            status = self._calculate_status(value, target)
            compliance = self._calculate_compliance(value, target)
            
            results[metric_name] = SLAMetric(
                name=target.name,
                current_value=value,
                target_value=target.target_value,
                compliance_percent=compliance,
                status=status,
                violation_duration_seconds=0,
                cost_impact=0
            )
        
        return results
    
    def analyze_throughput(self, requests_count: int, time_window_seconds: float) -> SLAMetric:
        throughput = requests_count / time_window_seconds if time_window_seconds > 0 else 0
        target = self.slo_targets["throughput"]
        status = self._calculate_status(throughput, target)
        compliance = self._calculate_compliance(throughput, target)
        
        return SLAMetric(
            name="Throughput",
            current_value=throughput,
            target_value=target.target_value,
            compliance_percent=compliance,
            status=status,
            violation_duration_seconds=0,
            cost_impact=0
        )
    
    def calculate_degradation_curve(self, attack_intensities: List[float], 
                                   response_times: List[float]) -> Dict:
        if len(attack_intensities) != len(response_times) or len(attack_intensities) < 3:
            return {"error": "insufficient_data"}
        
        x = np.array(attack_intensities)
        y = np.array(response_times)
        
        try:
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            
            target = self.slo_targets["latency_p95"]
            roots = np.roots([coeffs[0], coeffs[1], coeffs[2] - target.target_value])
            real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
            sla_breach_intensity = min(real_roots) if real_roots else None
            
            r_squared = 1 - (np.sum((y - poly(x))**2) / np.sum((y - np.mean(y))**2))
            
            return {
                "polynomial_coefficients": coeffs.tolist(),
                "r_squared": r_squared,
                "sla_breach_intensity": sla_breach_intensity,
                "degradation_rate": coeffs[1],
                "acceleration": coeffs[0],
                "predicted_values": poly(x).tolist()
            }
        except:
            return {"error": "calculation_failed"}
    
    def find_breaking_point(self, intensities: List[float], metrics: List[Dict]) -> Dict:
        if not intensities or not metrics:
            return {"breaking_point": None}
        
        for i, (intensity, metric) in enumerate(zip(intensities, metrics)):
            availability = metric.get("availability", 100)
            latency_p99 = metric.get("latency_p99", 0)
            
            avail_target = self.slo_targets["availability"]
            lat_target = self.slo_targets["latency_p99"]
            
            if availability < avail_target.critical_threshold or latency_p99 > lat_target.critical_threshold:
                return {
                    "breaking_point": intensity,
                    "breaking_index": i,
                    "reason": "availability" if availability < avail_target.critical_threshold else "latency",
                    "metrics_at_break": metric
                }
        
        return {"breaking_point": None, "reason": "no_breach_detected"}
    
    def generate_sla_report(self, metrics: Dict) -> Dict:
        total_requests = metrics.get("total_sent", 0)
        successful = metrics.get("total_received", 0) - metrics.get("total_blocked", 0)
        duration = metrics.get("duration_seconds", 1)
        latencies = metrics.get("latencies", [])
        
        availability = self.analyze_availability(total_requests, successful, duration)
        latency_metrics = self.analyze_latency(latencies)
        throughput = self.analyze_throughput(successful, duration)
        
        all_metrics = [availability, throughput] + list(latency_metrics.values())
        overall_compliance = np.mean([m.compliance_percent for m in all_metrics])
        
        violations = [m for m in all_metrics if m.status in [SLAStatus.VIOLATION, SLAStatus.CRITICAL]]
        total_cost = sum(m.cost_impact for m in all_metrics)
        
        return {
            "overall_compliance_percent": overall_compliance,
            "overall_status": SLAStatus.VIOLATION.value if violations else SLAStatus.COMPLIANT.value,
            "availability": {
                "value": availability.current_value,
                "target": availability.target_value,
                "status": availability.status.value,
                "compliance": availability.compliance_percent
            },
            "latency": {
                name: {
                    "value": m.current_value,
                    "target": m.target_value,
                    "status": m.status.value,
                    "compliance": m.compliance_percent
                } for name, m in latency_metrics.items()
            },
            "throughput": {
                "value": throughput.current_value,
                "target": throughput.target_value,
                "status": throughput.status.value,
                "compliance": throughput.compliance_percent
            },
            "violations_count": len(violations),
            "total_cost_impact": total_cost,
            "recommendations": self._generate_recommendations(all_metrics)
        }
    
    def _generate_recommendations(self, metrics: List[SLAMetric]) -> List[str]:
        recommendations = []
        
        for m in metrics:
            if m.status == SLAStatus.CRITICAL:
                if "Latency" in m.name:
                    recommendations.append(f"CRITICAL: {m.name} at {m.current_value:.2f}{self.slo_targets.get(m.name.lower().replace(' ', '_'), SLOTarget('', 0, 0, 0, '')).unit}. Consider scaling infrastructure or enabling caching.")
                elif "Availability" in m.name:
                    recommendations.append(f"CRITICAL: Availability at {m.current_value:.2f}%. Enable additional DDoS mitigation or increase rate limiting.")
            elif m.status == SLAStatus.VIOLATION:
                recommendations.append(f"WARNING: {m.name} violating SLA. Current: {m.current_value:.2f}, Target: {m.target_value:.2f}")
        
        return recommendations
