import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from .vrps import VRPSCalculator, VRPSConfig, VRPSVector, SustainabilityResult
from .lstm_predictor import LSTMPredictor, LSTMConfig, DataGenerator
from .kalman_filter import KalmanFilter, KalmanConfig, HybridPredictor
from .decision_matrix import (
    DecisionMatrix, DecisionConfig, DecisionResult,
    CosineSimilarity, SustainabilityIndex, StabilityRegion,
    ResponseMode, RESPONSE_ACTIONS
)


@dataclass
class MonitorConfig:
    vrps_config: Optional[VRPSConfig] = None
    lstm_config: Optional[LSTMConfig] = None
    lstm_model_path: Optional[str] = None
    kalman_config: Optional[KalmanConfig] = None
    decision_config: Optional[DecisionConfig] = None
    update_interval_seconds: float = 10.0
    history_size: int = 1000
    enable_lstm: bool = True
    enable_kalman: bool = True


@dataclass
class MonitoringSnapshot:
    timestamp: datetime
    vrps_vector: VRPSVector
    sustainability: SustainabilityResult
    prediction: Optional[np.ndarray]
    multi_step_predictions: List[np.ndarray]
    similarity: float
    similarity_ma: float
    decision: DecisionResult
    in_osr: bool
    osr_violations: List[str]
    model_version: str


class StabilityMonitor:
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.vrps = VRPSCalculator(self.config.vrps_config)
        self.lstm = None
        if self.config.enable_lstm:
            self.lstm = LSTMPredictor(self.config.lstm_config)
            if self.config.lstm_model_path:
                self.lstm.load_model(self.config.lstm_model_path)
        self.kalman = None
        self.hybrid = None
        if self.config.enable_kalman:
            self.kalman = KalmanFilter(self.config.kalman_config)
            self.hybrid = HybridPredictor(self.lstm, self.kalman)
        self.decision_matrix = DecisionMatrix(self.config.decision_config)
        self.stability_region = StabilityRegion()
        self._snapshots: List[MonitoringSnapshot] = []
        self._is_running = False
    
    def process_metrics(
        self,
        T: float,
        rho: float,
        P_block: float,
        U_cpu: float,
        U_ram: float,
        N_anom: int,
        N_bg: int,
        timestamp: Optional[datetime] = None
    ) -> MonitoringSnapshot:
        ts = timestamp or datetime.now()
        vrps_vector = self.vrps.calculate_vector(T, rho, P_block, U_cpu, U_ram, N_anom, N_bg, ts)
        sustainability = self.vrps.calculate_sustainability(vrps_vector)
        prediction = None
        multi_predictions = []
        history = self.vrps.get_history_array()
        seq_len = self.config.lstm_config.sequence_length if self.config.lstm_config else 15
        if len(history) >= seq_len:
            sequence = history[-seq_len:]
            if self.hybrid:
                result = self.hybrid.update_and_predict(vrps_vector.as_array, sequence, multi_step=5)
                prediction = result.get('lstm_prediction') or result.get('kalman_estimate')
                multi_predictions = result.get('multi_step_predictions', [])
            elif self.lstm and self.lstm.is_trained:
                pred_result = self.lstm.predict(sequence)
                prediction = pred_result.predicted_vector
        if prediction is None:
            prediction = vrps_vector.as_array
        sim_result = self.decision_matrix.similarity.evaluate(prediction, vrps_vector.as_array)
        decision = self.decision_matrix.decide(prediction, vrps_vector.as_array, self.stability_region.thresholds)
        in_osr = self.stability_region.is_in_osr(vrps_vector.as_array)
        violations = [v[0] for v in self.stability_region.get_violations(vrps_vector.as_array)]
        snapshot = MonitoringSnapshot(
            timestamp=ts,
            vrps_vector=vrps_vector,
            sustainability=sustainability,
            prediction=prediction,
            multi_step_predictions=multi_predictions,
            similarity=sim_result.similarity,
            similarity_ma=sim_result.moving_average,
            decision=decision,
            in_osr=in_osr,
            osr_violations=violations,
            model_version=self.lstm.model_version if self.lstm else "none"
        )
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.config.history_size:
            self._snapshots = self._snapshots[-self.config.history_size:]
        return snapshot

    def train_models(self, historical_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        results = {}
        if historical_data is None:
            print("Generating synthetic training data...")
            historical_data = DataGenerator.generate_mixed_dataset(1000)
        if self.lstm:
            print("Training LSTM model...")
            self.lstm.build_model()
            history = self.lstm.train(historical_data, verbose=1)
            results['lstm'] = {
                'trained': True,
                'history': history,
                'evaluation': self.lstm.evaluate(historical_data[-200:])
            }
        if self.kalman:
            print("Training Kalman transition matrix...")
            F = self.kalman.learn_transition_matrix(historical_data)
            results['kalman'] = {'trained': True, 'transition_matrix': F.tolist()}
        return results
    
    def get_current_status(self) -> Dict[str, Any]:
        if not self._snapshots:
            return {'status': 'no_data'}
        latest = self._snapshots[-1]
        return {
            'timestamp': latest.timestamp.isoformat(),
            'vrps': {
                'C': latest.vrps_vector.C_norm,
                'L': latest.vrps_vector.L_norm,
                'Q': latest.vrps_vector.Q_norm,
                'R': latest.vrps_vector.R_norm,
                'A': latest.vrps_vector.A_norm,
            },
            'sustainability': {
                'index': latest.sustainability.sust_index,
                'status': latest.sustainability.status.value,
            },
            'prediction': latest.prediction.tolist() if latest.prediction is not None else None,
            'similarity': latest.similarity,
            'similarity_ma': latest.similarity_ma,
            'decision': {
                'mode': latest.decision.mode.value,
                'action': latest.decision.action,
                'reason': latest.decision.reason,
            },
            'osr': {
                'in_osr': latest.in_osr,
                'violations': latest.osr_violations,
            }
        }
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        snapshots = self._snapshots[-last_n:] if last_n else self._snapshots
        return [
            {
                'timestamp': s.timestamp.isoformat(),
                'vrps': s.vrps_vector.as_array.tolist(),
                'sust_index': s.sustainability.sust_index,
                'similarity': s.similarity,
                'mode': s.decision.mode.value,
                'in_osr': s.in_osr,
            }
            for s in snapshots
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self._snapshots:
            return {}
        vrps_history = np.array([s.vrps_vector.as_array for s in self._snapshots])
        sust_history = [s.sustainability.sust_index for s in self._snapshots]
        sim_history = [s.similarity for s in self._snapshots]
        mode_stats = self.decision_matrix.get_mode_statistics()
        return {
            'total_snapshots': len(self._snapshots),
            'vrps_stats': {
                'mean': vrps_history.mean(axis=0).tolist(),
                'std': vrps_history.std(axis=0).tolist(),
                'min': vrps_history.min(axis=0).tolist(),
                'max': vrps_history.max(axis=0).tolist(),
            },
            'sustainability': {
                'mean': float(np.mean(sust_history)),
                'min': float(np.min(sust_history)),
                'max': float(np.max(sust_history)),
            },
            'similarity': {
                'mean': float(np.mean(sim_history)),
                'min': float(np.min(sim_history)),
            },
            'mode_distribution': {mode.name: count for mode, count in mode_stats.items()},
            'osr_violations_count': sum(1 for s in self._snapshots if not s.in_osr),
        }
    
    def export_for_dashboard(self) -> Dict[str, Any]:
        if not self._snapshots:
            return {}
        recent = self._snapshots[-100:]
        timestamps = [s.timestamp.isoformat() for s in recent]
        return {
            'timestamps': timestamps,
            'vrps_trajectory': {
                'C': [s.vrps_vector.C_norm for s in recent],
                'L': [s.vrps_vector.L_norm for s in recent],
                'Q': [s.vrps_vector.Q_norm for s in recent],
                'R': [s.vrps_vector.R_norm for s in recent],
                'A': [s.vrps_vector.A_norm for s in recent],
            },
            'sustainability_index': [s.sustainability.sust_index for s in recent],
            'similarity': [s.similarity for s in recent],
            'modes': [s.decision.mode.value for s in recent],
            'current': self.get_current_status(),
            'statistics': self.get_statistics(),
        }
    
    def save_state(self, path: str) -> None:
        state = {
            'snapshots_count': len(self._snapshots),
            'statistics': self.get_statistics(),
            'last_snapshot': self.get_current_status(),
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        if self.lstm and self.lstm.is_trained:
            self.lstm.save_model(path.replace('.json', '_lstm'))
    
    def reset(self) -> None:
        self._snapshots.clear()
        self.vrps.clear_history()
        if self.kalman:
            self.kalman.reset()


def create_monitor(
    enable_lstm: bool = True,
    enable_kalman: bool = True,
    lstm_sequence_length: int = 15
) -> StabilityMonitor:
    config = MonitorConfig(
        vrps_config=VRPSConfig(),
        lstm_config=LSTMConfig(sequence_length=lstm_sequence_length) if enable_lstm else None,
        kalman_config=KalmanConfig() if enable_kalman else None,
        decision_config=DecisionConfig(),
        enable_lstm=enable_lstm,
        enable_kalman=enable_kalman,
    )
    return StabilityMonitor(config)


def quick_demo():
    print("=== Stability Monitor Demo ===\n")
    monitor = create_monitor(enable_lstm=False, enable_kalman=True)
    print("Simulating normal operation...")
    for i in range(20):
        snapshot = monitor.process_metrics(
            T=15 + np.random.normal(0, 2),
            rho=0.3 + np.random.normal(0, 0.05),
            P_block=0.001 + np.random.normal(0, 0.0005),
            U_cpu=0.4 + np.random.normal(0, 0.05),
            U_ram=0.5 + np.random.normal(0, 0.05),
            N_anom=int(np.random.poisson(5)),
            N_bg=100
        )
    print(f"Status: {snapshot.sustainability.status.value}")
    print(f"Sust Index: {snapshot.sustainability.sust_index:.3f}")
    print(f"Mode: {snapshot.decision.mode.name}")
    print("\nSimulating attack...")
    for i in range(10):
        snapshot = monitor.process_metrics(
            T=200 + i * 30,
            rho=0.7 + i * 0.03,
            P_block=0.02 + i * 0.01,
            U_cpu=0.8 + i * 0.02,
            U_ram=0.7 + i * 0.02,
            N_anom=50 + i * 10,
            N_bg=100
        )
    print(f"Status: {snapshot.sustainability.status.value}")
    print(f"Sust Index: {snapshot.sustainability.sust_index:.3f}")
    print(f"Mode: {snapshot.decision.mode.name}")
    print(f"Action: {snapshot.decision.action}")
    print(f"Reason: {snapshot.decision.reason}")
    print("\n=== Statistics ===")
    stats = monitor.get_statistics()
    print(f"Total snapshots: {stats['total_snapshots']}")
    print(f"OSR violations: {stats['osr_violations_count']}")
    print(f"Mode distribution: {stats['mode_distribution']}")


if __name__ == "__main__":
    quick_demo()
