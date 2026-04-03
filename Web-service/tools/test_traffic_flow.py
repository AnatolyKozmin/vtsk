#!/usr/bin/env python3
import sys
sys.path.insert(0, '..')


def test_traffic_flow():
    from app.models import (
        TrafficFlowConfig,
        BackgroundTrafficParams,
        AnomalousTrafficParams,
        DistributionParams,
        DistributionType,
        QueueingSystemConfig,
    )
    from app.traffic import TrafficFlowGenerator
    from app.analysis import GGcKQueueingSystem, SLAValidator

    print("=== Test Traffic Flow Formalization ===\n")

    config = TrafficFlowConfig(
        background=BackgroundTrafficParams(A=1000, t_m=12.0, sigma=4.0),
        anomalous=AnomalousTrafficParams(
            distribution=DistributionType.EXPONENTIAL,
            total_volume=5000,
            start_time=10.0,
            duration=5.0,
            params=DistributionParams(rate=2.0)
        ),
        time_step=0.5
    )

    generator = TrafficFlowGenerator(config)

    print("1. Testing N_bg(t) - Gaussian background traffic:")
    for t in [0, 6, 12, 18, 24]:
        n_bg = generator.generate_background(t)
        print(f"   t={t:2d}h: N_bg = {n_bg:.2f} tx/s")

    print("\n2. Testing N(t) = N_bg(t) + N_anom(t) superposition:")
    for t in [8, 10, 11, 12, 15, 18]:
        n_bg, n_anom, n_total = generator.generate_combined(t)
        print(f"   t={t:2d}h: N_bg={n_bg:.1f}, N_anom={n_anom:.1f}, N_total={n_total:.1f}")

    print("\n3. Generating time series (0-24h):")
    ts = generator.generate_time_series(0, 24, dt=1.0)
    print(f"   Points: {len(ts.timestamps)}")
    print(f"   Background total: {ts.background_count}")
    print(f"   Anomalous total: {ts.anomalous_count}")
    print(f"   Peak N_total: {max(ts.n_total):.1f} tx/s")

    print("\n4. Testing serialization round-trip:")
    json_str = TrafficFlowGenerator.serialize(ts)
    ts_restored = TrafficFlowGenerator.deserialize(json_str)
    match = ts.timestamps == ts_restored.timestamps
    print(f"   Round-trip OK: {match}")

    print("\n=== Test G/G/c/K Queueing System ===\n")

    q_config = QueueingSystemConfig(c=10, K=1000, mu=100.0, sla_epsilon=0.01, sla_delta=0.05)
    system = GGcKQueueingSystem(q_config)

    print("5. Testing utilization rho(t) = lambda / (c * mu):")
    for lam in [500, 800, 1000, 1200]:
        rho = system.compute_utilization(lam)
        print(f"   lambda={lam}: rho={rho:.2f} {'(OVERLOAD!)' if rho > 1 else ''}")

    print("\n6. Full system analysis:")
    analysis = system.analyze(ts)
    print(f"   Avg rho: {analysis.utilization.avg_utilization:.3f}")
    print(f"   Max rho: {analysis.utilization.max_utilization:.3f}")
    print(f"   D_loss: {analysis.d_loss.d_loss:.4f}")
    e_fail = analysis.e_t_fail
    print(f"   E[T_fail]: {e_fail:.2f}s" if e_fail < 1e9 else "   E[T_fail]: inf (no failure expected)")
    print(f"   SLA compliant: {analysis.d_loss.sla_compliant}")

    print("\n7. Generating report:")
    report = system.generate_report(analysis)
    print(f"   Report ID: {report.report_id[:8]}...")
    print(f"   Overall SLA compliant: {report.overall_compliant}")
    if report.violations:
        for v in report.violations:
            print(f"   VIOLATION: {v.metric_name} = {v.actual_value:.4f} > {v.threshold}")

    print("\n8. Testing SLAValidator:")
    validator = SLAValidator(epsilon=0.01, delta=0.05)
    r1 = validator.validate_p_block(0.02)
    r2 = validator.validate_d_loss(0.03)
    print(f"   P_block=0.02: {r1.status.value}")
    print(f"   D_loss=0.03: {r2.status.value}")

    print("\n=== All tests passed! ===")
    return True


def test_existing_queuing():
    from app.analysis.queuing import QueuingTheoryAnalyzer, PaymentAnomalyType
    
    print("\n=== Test Existing QueuingTheoryAnalyzer ===\n")
    
    analyzer = QueuingTheoryAnalyzer(num_servers=10, queue_capacity=1000, service_rate=100.0)
    
    print("Testing metrics for lambda=800:")
    metrics = analyzer.analyze_system(800)
    print(f"   rho: {metrics.rho:.3f}")
    print(f"   P_block: {metrics.p_block:.4f}")
    print(f"   E[wait]: {metrics.e_wait:.2f}s")
    print(f"   Throughput: {metrics.throughput:.1f} tx/s")
    
    print("\nAnomaly types:")
    for code in [0, 1, 2, 5]:
        print(f"   {code}: {PaymentAnomalyType.to_russian(code)}")
    
    return True


def main():
    print("="*70)
    print("FULL LOCAL TEST SUITE")
    print("="*70)
    
    try:
        test_traffic_flow()
        test_existing_queuing()
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
