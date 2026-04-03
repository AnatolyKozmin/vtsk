import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

@dataclass
class AttackSignature:
    name: str
    pattern: Dict[str, float]
    confidence_threshold: float

class CorrelationAnalyzer:
    def __init__(self):
        self.attack_signatures = self._init_signatures()
        self.event_buffer: List[Dict] = []
        self.correlation_window = 60
    
    def _init_signatures(self) -> Dict[str, AttackSignature]:
        return {
            "syn_flood": AttackSignature(
                name="SYN Flood",
                pattern={
                    "request_rate_spike": 5.0,
                    "connection_ratio": 0.1,
                    "packet_size_variance": 0.2,
                    "source_diversity": 0.8
                },
                confidence_threshold=0.7
            ),
            "http_flood": AttackSignature(
                name="HTTP Flood",
                pattern={
                    "request_rate_spike": 3.0,
                    "response_time_increase": 2.0,
                    "error_rate_increase": 1.5,
                    "payload_similarity": 0.9
                },
                confidence_threshold=0.65
            ),
            "slowloris": AttackSignature(
                name="Slowloris",
                pattern={
                    "connection_duration_increase": 10.0,
                    "incomplete_requests": 0.8,
                    "low_bandwidth": 0.3,
                    "steady_connection_count": 0.9
                },
                confidence_threshold=0.75
            ),
            "amplification": AttackSignature(
                name="Amplification Attack",
                pattern={
                    "response_to_request_ratio": 10.0,
                    "udp_traffic_spike": 5.0,
                    "source_port_53_or_123": 0.7,
                    "spoofed_sources": 0.9
                },
                confidence_threshold=0.8
            ),
            "application_layer": AttackSignature(
                name="Application Layer Attack",
                pattern={
                    "specific_endpoint_targeting": 0.9,
                    "malformed_requests": 0.6,
                    "session_anomaly": 0.7,
                    "resource_exhaustion": 0.8
                },
                confidence_threshold=0.7
            )
        }
    
    def add_event(self, event: Dict):
        event["timestamp"] = datetime.now()
        self.event_buffer.append(event)
        
        cutoff = datetime.now().timestamp() - self.correlation_window
        self.event_buffer = [e for e in self.event_buffer 
                           if e["timestamp"].timestamp() > cutoff]
    
    def calculate_traffic_features(self, metrics: Dict) -> Dict[str, float]:
        baseline_rps = metrics.get("baseline_rps", 100)
        current_rps = metrics.get("current_rps", 100)
        baseline_latency = metrics.get("baseline_latency", 50)
        current_latency = metrics.get("current_latency", 50)
        baseline_error_rate = metrics.get("baseline_error_rate", 0.01)
        current_error_rate = metrics.get("current_error_rate", 0.01)
        
        return {
            "request_rate_spike": current_rps / baseline_rps if baseline_rps > 0 else 1.0,
            "response_time_increase": current_latency / baseline_latency if baseline_latency > 0 else 1.0,
            "error_rate_increase": current_error_rate / baseline_error_rate if baseline_error_rate > 0 else 1.0,
            "connection_ratio": metrics.get("active_connections", 0) / max(metrics.get("total_connections", 1), 1),
            "source_diversity": metrics.get("unique_sources", 1) / max(metrics.get("total_requests", 1), 1),
            "payload_similarity": metrics.get("payload_entropy", 0.5),
            "packet_size_variance": metrics.get("packet_size_std", 0) / max(metrics.get("packet_size_mean", 1), 1)
        }
    
    def match_signature(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        matches = []
        
        for sig_name, signature in self.attack_signatures.items():
            score = 0.0
            matched_features = 0
            
            for feature_name, threshold in signature.pattern.items():
                if feature_name in features:
                    feature_value = features[feature_name]
                    if feature_value >= threshold:
                        score += 1.0
                    else:
                        score += feature_value / threshold
                    matched_features += 1
            
            if matched_features > 0:
                confidence = score / matched_features
                if confidence >= signature.confidence_threshold:
                    matches.append((sig_name, confidence))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def correlate_events(self, events: List[Dict]) -> Dict:
        if not events:
            return {"correlations": [], "attack_timeline": []}
        
        event_types = defaultdict(list)
        for event in events:
            event_types[event.get("source", "unknown")].append(event)
        
        correlations = []
        
        sources = list(event_types.keys())
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                events1 = event_types[source1]
                events2 = event_types[source2]
                
                if len(events1) > 1 and len(events2) > 1:
                    times1 = [e["timestamp"].timestamp() for e in events1]
                    times2 = [e["timestamp"].timestamp() for e in events2]
                    
                    min_len = min(len(times1), len(times2))
                    if min_len >= 3:
                        corr, p_value = stats.pearsonr(times1[:min_len], times2[:min_len])
                        if abs(corr) > 0.5 and p_value < 0.05:
                            correlations.append({
                                "source1": source1,
                                "source2": source2,
                                "correlation": corr,
                                "p_value": p_value,
                                "interpretation": "strong_positive" if corr > 0.7 else "moderate"
                            })
        
        return {
            "correlations": correlations,
            "event_counts_by_source": {k: len(v) for k, v in event_types.items()},
            "total_events": len(events)
        }
    
    def detect_attack_start(self, time_series: List[Tuple[float, float]]) -> Optional[Dict]:
        if len(time_series) < 10:
            return None
        
        timestamps = [t[0] for t in time_series]
        values = [t[1] for t in time_series]
        
        window = 5
        for i in range(window, len(values)):
            baseline = np.mean(values[i-window:i])
            baseline_std = np.std(values[i-window:i])
            
            if baseline_std > 0:
                z_score = (values[i] - baseline) / baseline_std
                if z_score > 3:
                    return {
                        "attack_start_index": i,
                        "attack_start_time": timestamps[i],
                        "baseline_value": baseline,
                        "spike_value": values[i],
                        "z_score": z_score,
                        "confidence": min(1.0, z_score / 5)
                    }
        
        return None
    
    def classify_unknown_attack(self, features: Dict[str, float]) -> Dict:
        matches = self.match_signature(features)
        
        if matches:
            best_match, confidence = matches[0]
            return {
                "classification": best_match,
                "confidence": confidence,
                "alternative_matches": matches[1:3] if len(matches) > 1 else [],
                "features_analyzed": list(features.keys())
            }
        
        anomaly_score = sum(
            1 for v in features.values() 
            if isinstance(v, (int, float)) and v > 2.0
        ) / len(features)
        
        return {
            "classification": "unknown_anomaly" if anomaly_score > 0.3 else "normal",
            "confidence": anomaly_score,
            "features_analyzed": list(features.keys()),
            "recommendation": "Manual analysis required" if anomaly_score > 0.3 else "No action needed"
        }
    
    def analyze_combined_attack(self, events: List[Dict], metrics: Dict) -> Dict:
        features = self.calculate_traffic_features(metrics)
        signature_matches = self.match_signature(features)
        event_correlation = self.correlate_events(events)
        
        attack_types = [m[0] for m in signature_matches]
        is_combined = len(attack_types) > 1
        
        severity = "low"
        if len(signature_matches) >= 3:
            severity = "critical"
        elif len(signature_matches) >= 2:
            severity = "high"
        elif len(signature_matches) >= 1:
            severity = "medium"
        
        return {
            "is_combined_attack": is_combined,
            "detected_attack_types": attack_types,
            "signature_matches": [{"type": m[0], "confidence": m[1]} for m in signature_matches],
            "event_correlations": event_correlation["correlations"],
            "severity": severity,
            "recommended_actions": self._get_mitigation_recommendations(attack_types)
        }
    
    def _get_mitigation_recommendations(self, attack_types: List[str]) -> List[str]:
        recommendations = {
            "syn_flood": "Enable SYN cookies, increase backlog queue, deploy TCP proxy",
            "http_flood": "Enable rate limiting, deploy WAF rules, use CAPTCHA",
            "slowloris": "Reduce connection timeout, limit connections per IP, use reverse proxy",
            "amplification": "Block spoofed traffic, rate limit UDP, contact upstream provider",
            "application_layer": "Deploy application-specific WAF rules, enable bot detection"
        }
        
        return [recommendations.get(at, "Monitor and analyze traffic patterns") for at in attack_types]
