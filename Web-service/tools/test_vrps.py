#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime


def test_vrps_calculator():
    print("\n" + "="*60)
    print("TEST 1: VRPSCalculator")
    print("="*60)
    
    from app.analysis.vrps import VRPSCalculator, VRPSConfig
    
    config = VRPSConfig(T_base=10.0, T_crit=500.0, rho_thresh=0.85, P_thresh=0.05, U_crit=0.95)
    calc = VRPSCalculator(config)
    
    vector = calc.calculate_vector(T=20, rho=0.3, P_block=0.001, U_cpu=0.4, U_ram=0.5, N_anom=5, N_bg=100)
    
    print(f"\nНормальная работа:")
    print(f"  C_norm = {vector.C_norm:.3f}")
    print(f"  L_norm = {vector.L_norm:.3f}")
    print(f"  Q_norm = {vector.Q_norm:.3f}")
    print(f"  R_norm = {vector.R_norm:.3f}")
    print(f"  A_norm = {vector.A_norm:.3f}")
    
    sust = calc.calculate_sustainability(vector)
    print(f"\n  Sust Index = {sust.sust_index:.3f}")
    print(f"  Status = {sust.status.value}")
    print(f"  In OSR = {sust.in_osr}")
    
    vector2 = calc.calculate_vector(T=300, rho=0.9, P_block=0.08, U_cpu=0.92, U_ram=0.88, N_anom=40, N_bg=60)
    
    print(f"\nПод атакой:")
    print(f"  C_norm = {vector2.C_norm:.3f}")
    print(f"  L_norm = {vector2.L_norm:.3f}")
    print(f"  Q_norm = {vector2.Q_norm:.3f}")
    print(f"  R_norm = {vector2.R_norm:.3f}")
    print(f"  A_norm = {vector2.A_norm:.3f}")
    
    sust2 = calc.calculate_sustainability(vector2)
    print(f"\n  Sust Index = {sust2.sust_index:.3f}")
    print(f"  Status = {sust2.status.value}")
    print(f"  Violated = {sust2.violated_components}")
    
    print("\n✓ VRPSCalculator test passed")


def test_lstm_predictor():
    print("\n" + "="*60)
    print("TEST 2: LSTMPredictor")
    print("="*60)
    
    from app.analysis.lstm_predictor import LSTMPredictor, LSTMConfig, DataGenerator
    
    print("\nГенерация синтетических данных...")
    data = DataGenerator.generate_mixed_dataset(500)
    print(f"  Размер датасета: {data.shape}")
    
    config = LSTMConfig(sequence_length=10, epochs=5)
    predictor = LSTMPredictor(config)
    
    sequence = data[:10]
    result = predictor.predict(sequence)
    
    print(f"\nFallback prediction:")
    print(f"  Predicted: {result.predicted_vector}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Model: {result.model_version}")
    
    multi = predictor.predict_multi_step(sequence, steps=3)
    print(f"\nMulti-step predictions ({len(multi)} steps):")
    for i, pred in enumerate(multi):
        print(f"  Step {i+1}: {pred.round(3)}")
    
    print("\n✓ LSTMPredictor test passed")


def test_kalman_filter():
    print("\n" + "="*60)
    print("TEST 3: KalmanFilter")
    print("="*60)
    
    from app.analysis.kalman_filter import KalmanFilter, KalmanConfig
    
    config = KalmanConfig(n_states=5, process_noise=0.01, measurement_noise=0.05)
    kf = KalmanFilter(config)
    
    print("\nСимуляция фильтрации...")
    true_state = np.array([0.9, 0.85, 0.92, 0.88, 0.95])
    
    for i in range(10):
        noise = np.random.normal(0, 0.05, 5)
        measurement = true_state + noise
        measurement = np.clip(measurement, 0, 1)
        state = kf.update(measurement)
        if i % 3 == 0:
            print(f"  Step {i}: estimate = {state.x_est.round(3)}, innovation = {state.innovation:.4f}")
    
    predictions = kf.predict_multi_step(5)
    print(f"\nМногошаговый прогноз:")
    for i, pred in enumerate(predictions):
        print(f"  t+{i+1}: {pred.round(3)}")
    
    print("\nОбучение матрицы перехода...")
    from app.analysis.lstm_predictor import DataGenerator
    data = DataGenerator.generate_normal_scenario(100)
    F = kf.learn_transition_matrix(data)
    print(f"  F diagonal: {np.diag(F).round(3)}")
    
    print("\n✓ KalmanFilter test passed")


def test_decision_matrix():
    print("\n" + "="*60)
    print("TEST 4: DecisionMatrix")
    print("="*60)
    
    from app.analysis.decision_matrix import DecisionMatrix, CosineSimilarity, SustainabilityIndex, StabilityRegion
    
    dm = DecisionMatrix()
    
    print("\nКосинусное сходство:")
    s_pred = np.array([0.9, 0.85, 0.92, 0.88, 0.95])
    s_norm = np.array([0.88, 0.83, 0.90, 0.86, 0.93])
    
    sim = CosineSimilarity()
    result = sim.evaluate(s_pred, s_norm)
    print(f"  Similarity = {result.similarity:.4f}")
    print(f"  Level = {result.level.value}")
    print(f"  Alert = {result.alert}")
    
    print("\nИндекс устойчивости:")
    sust = SustainabilityIndex()
    index = sust.calculate(s_norm)
    level = sust.get_level(index)
    worst = sust.get_worst_component(s_norm)
    print(f"  Sust = {index:.4f}")
    print(f"  Level = {level.value}")
    print(f"  Worst component = {worst}")
    
    print("\nОбласть стабильной работы:")
    osr = StabilityRegion()
    in_osr = osr.is_in_osr(s_norm)
    violations = osr.get_violations(s_norm)
    distance = osr.distance_to_boundary(s_norm)
    print(f"  In OSR = {in_osr}")
    print(f"  Violations = {violations}")
    print(f"  Distance to boundary = {distance:.4f}")
    
    print("\nПринятие решения:")
    decision = dm.decide(s_pred, s_norm)
    print(f"  Mode = {decision.mode.name}")
    print(f"  Action = {decision.action}")
    print(f"  Reason = {decision.reason}")
    
    print("\nКритическое состояние:")
    s_critical = np.array([0.3, 0.2, 0.4, 0.5, 0.3])
    decision2 = dm.decide(s_critical, s_critical)
    print(f"  Mode = {decision2.mode.name}")
    print(f"  Action = {decision2.action}")
    
    print("\n✓ DecisionMatrix test passed")


def test_stability_monitor():
    print("\n" + "="*60)
    print("TEST 5: StabilityMonitor")
    print("="*60)
    
    from app.analysis.stability_monitor import create_monitor
    
    monitor = create_monitor(enable_lstm=False, enable_kalman=True)
    
    print("\nСимуляция нормальной работы (10 шагов)...")
    for i in range(10):
        snapshot = monitor.process_metrics(
            T=15 + np.random.normal(0, 2),
            rho=0.3 + np.random.normal(0, 0.05),
            P_block=0.001,
            U_cpu=0.4,
            U_ram=0.5,
            N_anom=3,
            N_bg=100
        )
    
    status = monitor.get_current_status()
    print(f"  Status: {status['sustainability']['status']}")
    print(f"  Sust: {status['sustainability']['index']:.3f}")
    print(f"  Mode: {status['decision']['mode']}")
    
    print("\nСимуляция атаки (10 шагов)...")
    for i in range(10):
        snapshot = monitor.process_metrics(
            T=100 + i * 40,
            rho=0.6 + i * 0.04,
            P_block=0.02 + i * 0.01,
            U_cpu=0.7 + i * 0.02,
            U_ram=0.65,
            N_anom=30 + i * 5,
            N_bg=70
        )
    
    status = monitor.get_current_status()
    print(f"  Status: {status['sustainability']['status']}")
    print(f"  Sust: {status['sustainability']['index']:.3f}")
    print(f"  Mode: {status['decision']['mode']}")
    print(f"  Action: {status['decision']['action']}")
    
    stats = monitor.get_statistics()
    print(f"\nСтатистика:")
    print(f"  Total snapshots: {stats['total_snapshots']}")
    print(f"  OSR violations: {stats['osr_violations_count']}")
    print(f"  Mode distribution: {stats['mode_distribution']}")
    
    print("\n✓ StabilityMonitor test passed")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           ТЕСТИРОВАНИЕ МОДУЛЕЙ ВРПС                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        test_vrps_calculator()
        test_lstm_predictor()
        test_kalman_filter()
        test_decision_matrix()
        test_stability_monitor()
        
        print("\n" + "="*60)
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
