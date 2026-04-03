import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime


@dataclass
class KalmanConfig:
    n_states: int = 5
    process_noise: float = 0.01
    measurement_noise: float = 0.05
    initial_covariance: float = 1.0


@dataclass
class KalmanState:
    x_est: np.ndarray
    P: np.ndarray
    timestamp: datetime
    innovation: float = 0.0


class KalmanFilter:
    def __init__(self, config: Optional[KalmanConfig] = None):
        self.config = config or KalmanConfig()
        n = self.config.n_states
        self.F = np.eye(n)
        self.H = np.eye(n)
        self.Q = np.eye(n) * self.config.process_noise
        self.R = np.eye(n) * self.config.measurement_noise
        self._x_est = np.ones(n) * 0.8
        self._P = np.eye(n) * self.config.initial_covariance
        self._history: List[KalmanState] = []
    
    def set_transition_matrix(self, F: np.ndarray) -> None:
        if F.shape != (self.config.n_states, self.config.n_states):
            raise ValueError(f"F должна быть {self.config.n_states}x{self.config.n_states}")
        self.F = F.copy()
    
    def learn_transition_matrix(self, data: np.ndarray) -> np.ndarray:
        if len(data) < 10:
            return self.F
        X = data[:-1].T
        Y = data[1:].T
        try:
            XXT_inv = np.linalg.inv(X @ X.T + np.eye(5) * 1e-6)
            self.F = Y @ X.T @ XXT_inv
        except np.linalg.LinAlgError:
            pass
        return self.F

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        x_pred = self.F @ self._x_est
        P_pred = self.F @ self._P @ self.F.T + self.Q
        return x_pred, P_pred
    
    def update(self, z: np.ndarray) -> KalmanState:
        x_pred, P_pred = self.predict()
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self._x_est = x_pred + K @ y
        I = np.eye(self.config.n_states)
        self._P = (I - K @ self.H) @ P_pred
        self._x_est = np.clip(self._x_est, 0, 1)
        state = KalmanState(
            x_est=self._x_est.copy(),
            P=self._P.copy(),
            timestamp=datetime.now(),
            innovation=float(np.linalg.norm(y))
        )
        self._history.append(state)
        return state
    
    def predict_multi_step(self, steps: int) -> List[np.ndarray]:
        predictions = []
        F_power = np.eye(self.config.n_states)
        for _ in range(steps):
            F_power = F_power @ self.F
            pred = F_power @ self._x_est
            pred = np.clip(pred, 0, 1)
            predictions.append(pred)
        return predictions
    
    def get_current_estimate(self) -> np.ndarray:
        return self._x_est.copy()
    
    def get_uncertainty(self) -> np.ndarray:
        return np.diag(self._P)
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        n = self.config.n_states
        self._x_est = initial_state if initial_state is not None else np.ones(n) * 0.8
        self._P = np.eye(n) * self.config.initial_covariance
        self._history.clear()
    
    def get_history(self, last_n: Optional[int] = None) -> List[KalmanState]:
        if last_n:
            return self._history[-last_n:]
        return self._history.copy()


class HybridPredictor:
    def __init__(self, lstm_predictor=None, kalman_filter: Optional[KalmanFilter] = None):
        self.lstm = lstm_predictor
        self.kalman = kalman_filter or KalmanFilter()
        self._last_lstm_prediction: Optional[np.ndarray] = None
    
    def update_and_predict(
        self,
        current_measurement: np.ndarray,
        lstm_sequence: Optional[np.ndarray] = None,
        multi_step: int = 5
    ) -> dict:
        kalman_state = self.kalman.update(current_measurement)
        lstm_pred = None
        if self.lstm is not None and lstm_sequence is not None:
            try:
                result = self.lstm.predict(lstm_sequence)
                lstm_pred = result.predicted_vector
                self._last_lstm_prediction = lstm_pred
            except Exception:
                pass
        multi_predictions = self.kalman.predict_multi_step(multi_step)
        return {
            'kalman_estimate': kalman_state.x_est,
            'lstm_prediction': lstm_pred,
            'multi_step_predictions': multi_predictions,
            'uncertainty': self.kalman.get_uncertainty(),
            'innovation': kalman_state.innovation
        }
    
    def train_kalman_from_data(self, historical_data: np.ndarray) -> None:
        self.kalman.learn_transition_matrix(historical_data)
