import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime
import json
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class LSTMConfig:
    sequence_length: int = 15
    n_features: int = 5
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 150
    batch_size: int = 32
    validation_split: float = 0.2


@dataclass
class PredictionResult:
    timestamp: datetime
    predicted_vector: np.ndarray
    confidence: float
    model_version: str


class LSTMPredictor:
    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.model = None
        self.is_trained = False
        self.training_history = None
        self.model_version = "v1.0"
        if not TF_AVAILABLE:
            print("WARNING: TensorFlow not available. Using fallback predictor.")
    
    def build_model(self) -> None:
        if not TF_AVAILABLE:
            return
        cfg = self.config
        self.model = keras.Sequential([
            layers.LSTM(cfg.lstm_units_1, return_sequences=True,
                       input_shape=(cfg.sequence_length, cfg.n_features), name='lstm_1'),
            layers.LSTM(cfg.lstm_units_2, return_sequences=False, name='lstm_2'),
            layers.Dropout(cfg.dropout_rate, name='dropout'),
            layers.Dense(cfg.n_features, activation='sigmoid', name='output')
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss='mse',
            metrics=['mae']
        )

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = self.config.sequence_length
        n_samples = len(data) - k
        if n_samples <= 0:
            raise ValueError(f"Недостаточно данных. Нужно минимум {k + 1} точек.")
        X = np.zeros((n_samples, k, self.config.n_features))
        y = np.zeros((n_samples, self.config.n_features))
        for i in range(n_samples):
            X[i] = data[i:i + k]
            y[i] = data[i + k]
        return X, y
    
    def train(self, data: np.ndarray, verbose: int = 1) -> dict:
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping training.")
            return {}
        if self.model is None:
            self.build_model()
        X, y = self.prepare_sequences(data)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        self.is_trained = True
        self.training_history = history.history
        self.model_version = f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return history.history
    
    def predict(self, sequence: np.ndarray) -> PredictionResult:
        if not TF_AVAILABLE or not self.is_trained:
            return self._fallback_predict(sequence)
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        prediction = self.model.predict(sequence, verbose=0)[0]
        prediction = np.clip(prediction, 0, 1)
        return PredictionResult(
            timestamp=datetime.now(),
            predicted_vector=prediction,
            confidence=self._estimate_confidence(sequence, prediction),
            model_version=self.model_version
        )
    
    def predict_multi_step(self, sequence: np.ndarray, steps: int = 5) -> List[np.ndarray]:
        predictions = []
        current_seq = sequence.copy()
        for _ in range(steps):
            result = self.predict(current_seq)
            predictions.append(result.predicted_vector)
            current_seq = np.vstack([current_seq[1:], result.predicted_vector])
        return predictions
    
    def _fallback_predict(self, sequence: np.ndarray) -> PredictionResult:
        if sequence.ndim == 3:
            sequence = sequence[0]
        weights = np.exp(np.linspace(-1, 0, len(sequence)))
        weights /= weights.sum()
        prediction = np.average(sequence, axis=0, weights=weights)
        return PredictionResult(
            timestamp=datetime.now(),
            predicted_vector=prediction,
            confidence=0.5,
            model_version="fallback_ema"
        )
    
    def _estimate_confidence(self, sequence: np.ndarray, prediction: np.ndarray) -> float:
        if sequence.ndim == 3:
            sequence = sequence[0]
        std = np.std(sequence, axis=0)
        mean_std = np.mean(std)
        confidence = max(0.3, 1 - mean_std)
        return min(0.99, confidence)

    def save_model(self, path: str) -> None:
        if not TF_AVAILABLE or self.model is None:
            return
        self.model.save(path)
        config_path = path + '_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'sequence_length': self.config.sequence_length,
                'n_features': self.config.n_features,
                'lstm_units_1': self.config.lstm_units_1,
                'lstm_units_2': self.config.lstm_units_2,
                'model_version': self.model_version,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self, path: str) -> bool:
        if not TF_AVAILABLE:
            return False
        try:
            self.model = keras.models.load_model(path)
            config_path = path + '_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    self.model_version = cfg.get('model_version', 'loaded')
                    self.is_trained = cfg.get('is_trained', True)
            else:
                self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate(self, data: np.ndarray) -> dict:
        if not TF_AVAILABLE or not self.is_trained:
            return {'mse': None, 'mae': None, 'r2': None}
        X, y_true = self.prepare_sequences(data)
        y_pred = self.model.predict(X, verbose=0)
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return {'mse': float(mse), 'mae': float(mae), 'r2': float(r2), 'samples': len(X)}


class DataGenerator:
    @staticmethod
    def generate_normal_scenario(n_steps: int = 1000, noise: float = 0.05) -> np.ndarray:
        t = np.linspace(0, 10 * np.pi, n_steps)
        C = 0.9 + 0.05 * np.sin(t * 0.5) + np.random.normal(0, noise, n_steps)
        L = 0.85 + 0.08 * np.sin(t * 0.3) + np.random.normal(0, noise, n_steps)
        Q = 0.92 + 0.04 * np.sin(t * 0.7) + np.random.normal(0, noise, n_steps)
        R = 0.88 + 0.06 * np.sin(t * 0.4) + np.random.normal(0, noise, n_steps)
        A = 0.95 + 0.03 * np.sin(t * 0.2) + np.random.normal(0, noise, n_steps)
        data = np.column_stack([C, L, Q, R, A])
        return np.clip(data, 0, 1)
    
    @staticmethod
    def generate_ddos_scenario(n_steps: int = 1000) -> np.ndarray:
        data = DataGenerator.generate_normal_scenario(n_steps)
        attack_start = int(n_steps * 0.4)
        attack_end = int(n_steps * 0.7)
        attack_profile = np.zeros(n_steps)
        attack_profile[attack_start:attack_end] = np.linspace(0, 0.6, attack_end - attack_start)
        attack_profile[attack_end:] = np.linspace(0.6, 0, n_steps - attack_end)
        data[:, 0] -= attack_profile * 0.5
        data[:, 1] -= attack_profile * 0.7
        data[:, 2] -= attack_profile * 0.6
        data[:, 4] -= attack_profile * 0.8
        return np.clip(data, 0, 1)
    
    @staticmethod
    def generate_slow_attack_scenario(n_steps: int = 1000) -> np.ndarray:
        data = DataGenerator.generate_normal_scenario(n_steps)
        degradation = np.linspace(0, 0.4, n_steps)
        data[:, 0] -= degradation * 0.3
        data[:, 1] -= degradation * 0.5
        data[:, 3] -= degradation * 0.4
        return np.clip(data, 0, 1)
    
    @staticmethod
    def generate_load_spike_scenario(n_steps: int = 1000) -> np.ndarray:
        data = DataGenerator.generate_normal_scenario(n_steps)
        for spike_center in [200, 500, 800]:
            spike = np.exp(-((np.arange(n_steps) - spike_center) ** 2) / 1000)
            data[:, 1] -= spike * 0.4
            data[:, 2] -= spike * 0.3
        return np.clip(data, 0, 1)
    
    @staticmethod
    def generate_mixed_dataset(n_steps_per_scenario: int = 1000) -> np.ndarray:
        scenarios = [
            DataGenerator.generate_normal_scenario(n_steps_per_scenario),
            DataGenerator.generate_ddos_scenario(n_steps_per_scenario),
            DataGenerator.generate_slow_attack_scenario(n_steps_per_scenario),
            DataGenerator.generate_load_spike_scenario(n_steps_per_scenario),
        ]
        return np.vstack(scenarios)
