import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import deque


class ResponseMode(Enum):
    MODE_1_MONITORING = 1
    MODE_2_ENHANCED_MONITORING = 2
    MODE_3_MODEL_ANALYSIS = 3
    MODE_4_TARGET_MITIGATION = 4
    MODE_5_GENERAL_PROTECTION = 5
    MODE_6_PREVENTIVE_ALERT = 6
    MODE_7_AGGRESSIVE_MITIGATION = 7
    MODE_8_COMPREHENSIVE_DEFENSE = 8
    MODE_9_CRITICAL_LOCKDOWN = 9


class SustainabilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SimilarityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


RESPONSE_ACTIONS = {
    ResponseMode.MODE_1_MONITORING: "pass",
    ResponseMode.MODE_2_ENHANCED_MONITORING: "log_enhanced",
    ResponseMode.MODE_3_MODEL_ANALYSIS: "alert_model_drift",
    ResponseMode.MODE_4_TARGET_MITIGATION: "target_mitigation",
    ResponseMode.MODE_5_GENERAL_PROTECTION: "general_protection",
    ResponseMode.MODE_6_PREVENTIVE_ALERT: "preventive_alert",
    ResponseMode.MODE_7_AGGRESSIVE_MITIGATION: "aggressive_mitigation",
    ResponseMode.MODE_8_COMPREHENSIVE_DEFENSE: "comprehensive_defense",
    ResponseMode.MODE_9_CRITICAL_LOCKDOWN: "critical_lockdown",
}


@dataclass
class SimilarityResult:
    timestamp: datetime
    similarity: float
    level: SimilarityLevel
    moving_average: float
    needs_retraining: bool
    alert: bool


@dataclass
class DecisionResult:
    timestamp: datetime
    mode: ResponseMode
    action: str
    sust_level: SustainabilityLevel
    sim_level: SimilarityLevel
    sust_value: float
    sim_value: float
    reason: str
    violated_components: List[str]


@dataclass
class DecisionConfig:
    sim_high_threshold: float = 0.9
    sim_medium_threshold: float = 0.6
    sim_alert_threshold: float = 0.7
    sim_retrain_threshold: float = 0.6
    sust_high_threshold: float = 0.8
    sust_medium_threshold: float = 0.5
    ma_window_size: int = 10
    retrain_duration_minutes: int = 5
    weights: Dict[str, float] = field(default_factory=lambda: {
        'C': 0.25, 'L': 0.20, 'Q': 0.35, 'R': 0.10, 'A': 0.10
    })


class CosineSimilarity:
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
        self._history: deque = deque(maxlen=1000)
        self._low_sim_start: Optional[datetime] = None
    
    def calculate(self, s_pred: np.ndarray, s_norm: np.ndarray) -> float:
        norm_pred = np.linalg.norm(s_pred)
        norm_real = np.linalg.norm(s_norm)
        if norm_pred < 1e-10 or norm_real < 1e-10:
            return 0.0
        dot_product = np.dot(s_pred, s_norm)
        similarity = dot_product / (norm_pred * norm_real)
        return float(np.clip(similarity, -1, 1))
    
    def evaluate(self, s_pred: np.ndarray, s_norm: np.ndarray) -> SimilarityResult:
        sim = self.calculate(s_pred, s_norm)
        now = datetime.now()
        self._history.append((now, sim))
        if sim > self.config.sim_high_threshold:
            level = SimilarityLevel.HIGH
        elif sim >= self.config.sim_medium_threshold:
            level = SimilarityLevel.MEDIUM
        else:
            level = SimilarityLevel.LOW
        ma = self._calculate_moving_average()
        needs_retraining = self._check_retrain_trigger(sim, now)
        alert = sim < self.config.sim_alert_threshold
        return SimilarityResult(
            timestamp=now, similarity=sim, level=level,
            moving_average=ma, needs_retraining=needs_retraining, alert=alert
        )
    
    def _calculate_moving_average(self) -> float:
        if not self._history:
            return 1.0
        window = list(self._history)[-self.config.ma_window_size:]
        values = [v for _, v in window]
        return float(np.mean(values))
    
    def _check_retrain_trigger(self, current_sim: float, now: datetime) -> bool:
        ma = self._calculate_moving_average()
        if ma < self.config.sim_retrain_threshold:
            if self._low_sim_start is None:
                self._low_sim_start = now
            elif (now - self._low_sim_start).total_seconds() > self.config.retrain_duration_minutes * 60:
                return True
        else:
            self._low_sim_start = None
        return False
    
    def get_history(self, last_n: Optional[int] = None) -> List[Tuple[datetime, float]]:
        history = list(self._history)
        if last_n:
            return history[-last_n:]
        return history


class SustainabilityIndex:
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
    
    def calculate(self, s_norm: np.ndarray) -> float:
        weights = self.config.weights
        w = np.array([weights['C'], weights['L'], weights['Q'], weights['R'], weights['A']])
        sust = np.dot(w, s_norm)
        return float(np.clip(sust, 0, 1))
    
    def get_level(self, sust: float) -> SustainabilityLevel:
        if sust > self.config.sust_high_threshold:
            return SustainabilityLevel.HIGH
        elif sust >= self.config.sust_medium_threshold:
            return SustainabilityLevel.MEDIUM
        else:
            return SustainabilityLevel.LOW
    
    def get_worst_component(self, s_norm: np.ndarray) -> Tuple[str, float]:
        components = ['C', 'L', 'Q', 'R', 'A']
        min_idx = np.argmin(s_norm)
        return components[min_idx], float(s_norm[min_idx])


class DecisionMatrix:
    DECISION_MATRIX = {
        SustainabilityLevel.HIGH: {
            SimilarityLevel.HIGH: ResponseMode.MODE_1_MONITORING,
            SimilarityLevel.MEDIUM: ResponseMode.MODE_2_ENHANCED_MONITORING,
            SimilarityLevel.LOW: ResponseMode.MODE_3_MODEL_ANALYSIS,
        },
        SustainabilityLevel.MEDIUM: {
            SimilarityLevel.HIGH: ResponseMode.MODE_4_TARGET_MITIGATION,
            SimilarityLevel.MEDIUM: ResponseMode.MODE_5_GENERAL_PROTECTION,
            SimilarityLevel.LOW: ResponseMode.MODE_6_PREVENTIVE_ALERT,
        },
        SustainabilityLevel.LOW: {
            SimilarityLevel.HIGH: ResponseMode.MODE_7_AGGRESSIVE_MITIGATION,
            SimilarityLevel.MEDIUM: ResponseMode.MODE_8_COMPREHENSIVE_DEFENSE,
            SimilarityLevel.LOW: ResponseMode.MODE_9_CRITICAL_LOCKDOWN,
        },
    }
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
        self.similarity = CosineSimilarity(config)
        self.sustainability = SustainabilityIndex(config)
        self._decision_history: List[DecisionResult] = []
    
    def decide(
        self,
        s_pred: np.ndarray,
        s_norm: np.ndarray,
        osr_thresholds: Optional[Tuple[float, ...]] = None
    ) -> DecisionResult:
        now = datetime.now()
        sim_result = self.similarity.evaluate(s_pred, s_norm)
        sust_pred = self.sustainability.calculate(s_pred)
        sust_level = self.sustainability.get_level(sust_pred)
        mode = self.DECISION_MATRIX[sust_level][sim_result.level]
        action = RESPONSE_ACTIONS[mode]
        violated = self._check_violations(s_norm, osr_thresholds)
        reason = self._generate_reason(sust_level, sim_result.level, violated)
        result = DecisionResult(
            timestamp=now, mode=mode, action=action,
            sust_level=sust_level, sim_level=sim_result.level,
            sust_value=sust_pred, sim_value=sim_result.similarity,
            reason=reason, violated_components=violated
        )
        self._decision_history.append(result)
        return result
    
    def _check_violations(
        self,
        s_norm: np.ndarray,
        osr_thresholds: Optional[Tuple[float, ...]] = None
    ) -> List[str]:
        if osr_thresholds is None:
            osr_thresholds = (0.7, 0.7, 0.8, 0.6, 0.9)
        components = ['C', 'L', 'Q', 'R', 'A']
        violated = []
        for i, (val, thresh) in enumerate(zip(s_norm, osr_thresholds)):
            if val < thresh:
                violated.append(components[i])
        return violated
    
    def _generate_reason(
        self,
        sust_level: SustainabilityLevel,
        sim_level: SimilarityLevel,
        violated: List[str]
    ) -> str:
        reasons = []
        if sust_level == SustainabilityLevel.LOW:
            reasons.append("Критически низкий индекс устойчивости")
        elif sust_level == SustainabilityLevel.MEDIUM:
            reasons.append("Умеренная деградация системы")
        if sim_level == SimilarityLevel.LOW:
            reasons.append("Низкое качество прогноза (возможен дрейф модели)")
        elif sim_level == SimilarityLevel.MEDIUM:
            reasons.append("Прогноз требует внимания")
        if violated:
            reasons.append(f"Нарушены компоненты ОСР: {', '.join(violated)}")
        return "; ".join(reasons) if reasons else "Система в норме"
    
    def get_decision_history(self, last_n: Optional[int] = None) -> List[DecisionResult]:
        if last_n:
            return self._decision_history[-last_n:]
        return self._decision_history.copy()
    
    def get_mode_statistics(self) -> Dict[ResponseMode, int]:
        stats = {mode: 0 for mode in ResponseMode}
        for decision in self._decision_history:
            stats[decision.mode] += 1
        return stats


class StabilityRegion:
    def __init__(self, thresholds: Optional[Tuple[float, ...]] = None):
        self.thresholds = thresholds or (0.7, 0.7, 0.8, 0.6, 0.9)
        self.component_names = ['C', 'L', 'Q', 'R', 'A']
    
    def is_in_osr(self, s_norm: np.ndarray) -> bool:
        return all(val >= thresh for val, thresh in zip(s_norm, self.thresholds))
    
    def get_violations(self, s_norm: np.ndarray) -> List[Tuple[str, float, float]]:
        violations = []
        for name, val, thresh in zip(self.component_names, s_norm, self.thresholds):
            if val < thresh:
                violations.append((name, float(val), thresh))
        return violations
    
    def distance_to_boundary(self, s_norm: np.ndarray) -> float:
        margins = [val - thresh for val, thresh in zip(s_norm, self.thresholds)]
        return float(min(margins))
    
    def will_exit_osr(self, s_pred: np.ndarray) -> Tuple[bool, List[str]]:
        violations = self.get_violations(s_pred)
        will_exit = len(violations) > 0
        violated_components = [v[0] for v in violations]
        return will_exit, violated_components
