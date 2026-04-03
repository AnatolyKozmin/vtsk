import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from enum import Enum


class SystemStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class VRPSConfig:
    T_base: float = 10.0
    T_crit: float = 500.0
    rho_thresh: float = 0.85
    P_thresh: float = 0.05
    U_crit: float = 0.95
    w_c: float = 0.25
    w_l: float = 0.20
    w_q: float = 0.35
    w_r: float = 0.10
    w_a: float = 0.10
    osr_thresholds: Tuple[float, ...] = (0.7, 0.7, 0.8, 0.6, 0.9)


@dataclass
class VRPSVector:
    timestamp: datetime
    C_norm: float
    L_norm: float
    Q_norm: float
    R_norm: float
    A_norm: float
    T_raw: float = 0.0
    rho_raw: float = 0.0
    P_block_raw: float = 0.0
    U_raw: float = 0.0
    alpha_raw: float = 0.0

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.C_norm, self.L_norm, self.Q_norm, self.R_norm, self.A_norm])
    
    @property
    def sustainability_index(self) -> float:
        return np.mean(self.as_array)


@dataclass
class SustainabilityResult:
    timestamp: datetime
    sust_index: float
    status: SystemStatus
    vector: VRPSVector
    in_osr: bool
    violated_components: List[str]


class VRPSCalculator:
    def __init__(self, config: Optional[VRPSConfig] = None):
        self.config = config or VRPSConfig()
        self._history: List[VRPSVector] = []
    
    def calculate_C_norm(self, T: float) -> float:
        if T <= self.config.T_base:
            return 1.0
        c_norm = 1 - (T - self.config.T_base) / (self.config.T_crit - self.config.T_base)
        return max(0.0, min(1.0, c_norm))
    
    def calculate_L_norm(self, rho: float) -> float:
        if rho <= 0:
            return 1.0
        l_norm = 1 - (rho / self.config.rho_thresh) ** 2
        return max(0.0, min(1.0, l_norm))
    
    def calculate_Q_norm(self, P_block: float) -> float:
        if P_block <= 0:
            return 1.0
        q_norm = 1 - P_block / self.config.P_thresh
        return max(0.0, min(1.0, q_norm))
    
    def calculate_R_norm(self, U_cpu: float, U_ram: float) -> float:
        U = max(U_cpu, U_ram)
        if U <= 0:
            return 1.0
        r_norm = 1 - U / self.config.U_crit
        return max(0.0, min(1.0, r_norm))
    
    def calculate_A_norm(self, N_anom: int, N_bg: int) -> float:
        total = N_anom + N_bg
        if total <= 0:
            return 1.0
        alpha = N_anom / total
        return 1 - alpha

    def calculate_vector(
        self,
        T: float,
        rho: float,
        P_block: float,
        U_cpu: float,
        U_ram: float,
        N_anom: int,
        N_bg: int,
        timestamp: Optional[datetime] = None
    ) -> VRPSVector:
        ts = timestamp or datetime.now()
        vector = VRPSVector(
            timestamp=ts,
            C_norm=self.calculate_C_norm(T),
            L_norm=self.calculate_L_norm(rho),
            Q_norm=self.calculate_Q_norm(P_block),
            R_norm=self.calculate_R_norm(U_cpu, U_ram),
            A_norm=self.calculate_A_norm(N_anom, N_bg),
            T_raw=T,
            rho_raw=rho,
            P_block_raw=P_block,
            U_raw=max(U_cpu, U_ram),
            alpha_raw=N_anom / (N_anom + N_bg) if (N_anom + N_bg) > 0 else 0
        )
        self._history.append(vector)
        return vector
    
    def calculate_sustainability(self, vector: VRPSVector) -> SustainabilityResult:
        cfg = self.config
        sust = (
            cfg.w_c * vector.C_norm +
            cfg.w_l * vector.L_norm +
            cfg.w_q * vector.Q_norm +
            cfg.w_r * vector.R_norm +
            cfg.w_a * vector.A_norm
        )
        if sust > 0.8:
            status = SystemStatus.HEALTHY
        elif sust >= 0.5:
            status = SystemStatus.DEGRADED
        else:
            status = SystemStatus.CRITICAL
        in_osr, violated = self._check_osr(vector)
        return SustainabilityResult(
            timestamp=vector.timestamp,
            sust_index=sust,
            status=status,
            vector=vector,
            in_osr=in_osr,
            violated_components=violated
        )
    
    def _check_osr(self, vector: VRPSVector) -> Tuple[bool, List[str]]:
        thresholds = self.config.osr_thresholds
        components = ['C', 'L', 'Q', 'R', 'A']
        values = vector.as_array
        violated = []
        for i, (val, thresh, name) in enumerate(zip(values, thresholds, components)):
            if val < thresh:
                violated.append(name)
        return len(violated) == 0, violated
    
    def get_history(self, last_n: Optional[int] = None) -> List[VRPSVector]:
        if last_n:
            return self._history[-last_n:]
        return self._history.copy()
    
    def get_history_array(self, last_n: Optional[int] = None) -> np.ndarray:
        history = self.get_history(last_n)
        if not history:
            return np.array([]).reshape(0, 5)
        return np.array([v.as_array for v in history])
    
    def clear_history(self):
        self._history.clear()


class MetricsCollector:
    def __init__(self, vrps_calculator: Optional[VRPSCalculator] = None):
        self.vrps = vrps_calculator or VRPSCalculator()
        self._transaction_times: List[float] = []
        self._blocked_count: int = 0
        self._total_count: int = 0
        self._anomaly_count: int = 0
        self._normal_count: int = 0
    
    def record_transaction(self, processing_time_ms: float, is_blocked: bool, is_anomaly: bool):
        self._transaction_times.append(processing_time_ms)
        self._total_count += 1
        if is_blocked:
            self._blocked_count += 1
        if is_anomaly:
            self._anomaly_count += 1
        else:
            self._normal_count += 1
    
    def get_current_metrics(self) -> Dict[str, float]:
        if not self._transaction_times:
            return {'T_avg': 0, 'P_block': 0, 'N_anom': 0, 'N_bg': 0}
        return {
            'T_avg': np.mean(self._transaction_times[-100:]),
            'P_block': self._blocked_count / self._total_count if self._total_count > 0 else 0,
            'N_anom': self._anomaly_count,
            'N_bg': self._normal_count
        }
    
    def calculate_rho(self, lambda_rate: float, c: int, mu: float) -> float:
        if c * mu <= 0:
            return 1.0
        return lambda_rate / (c * mu)
    
    def reset_window(self):
        self._transaction_times.clear()
        self._blocked_count = 0
        self._total_count = 0
        self._anomaly_count = 0
        self._normal_count = 0
