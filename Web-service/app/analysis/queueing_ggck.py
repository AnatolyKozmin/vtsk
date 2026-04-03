import math
from typing import List, Tuple, Optional
from scipy.special import factorial
import numpy as np
from app.models.queueing import (
    QueueingSystemConfig,
    UtilizationResult,
    DLossResult,
    BlockingEvent,
    QueueingMetricsSnapshot,
    QueueingAnalysisResult,
    SLAComplianceResult,
    SLAComplianceStatus,
    SLAViolation,
    SystemReport,
    ComparisonReport,
)
from app.models.traffic_flow import TrafficTimeSeries
from datetime import datetime
import uuid


class GGcKQueueingSystem:
    def __init__(self, config: QueueingSystemConfig):
        self.config = config
        self.c = config.c
        self.K = config.K
        self.mu = config.mu
        self._blocking_events: List[BlockingEvent] = []

    def compute_utilization(self, lambda_t: float) -> float:
        if self.mu <= 0:
            raise ValueError("Service rate mu must be positive")
        return lambda_t / (self.c * self.mu)

    def compute_utilization_series(
        self, lambda_series: List[Tuple[float, float]], window_size: int = 10
    ) -> UtilizationResult:
        timestamps = []
        instantaneous = []
        moving_avg = []
        overload_periods = []
        in_overload = False
        overload_start = 0.0
        for t, lam in lambda_series:
            rho = self.compute_utilization(lam)
            timestamps.append(t)
            instantaneous.append(rho)
            if rho > 1.0:
                if not in_overload:
                    in_overload = True
                    overload_start = t
            else:
                if in_overload:
                    overload_periods.append((overload_start, t))
                    in_overload = False
        if in_overload and timestamps:
            overload_periods.append((overload_start, timestamps[-1]))
        for i in range(len(instantaneous)):
            start_idx = max(0, i - window_size + 1)
            window = instantaneous[start_idx : i + 1]
            moving_avg.append(sum(window) / len(window))
        max_util = max(instantaneous) if instantaneous else 0.0
        avg_util = sum(instantaneous) / len(instantaneous) if instantaneous else 0.0
        return UtilizationResult(
            timestamps=timestamps,
            instantaneous=instantaneous,
            moving_average=moving_avg,
            overload_periods=overload_periods,
            max_utilization=max_util,
            avg_utilization=avg_util,
        )


    def erlang_b(self, a: float) -> float:
        if a <= 0:
            return 0.0
        c = self.c
        numerator = (a ** c) / factorial(c)
        denominator = sum((a ** n) / factorial(n) for n in range(c + 1))
        return numerator / denominator if denominator > 0 else 0.0

    def compute_p_block_stationary(self, lambda_t: float) -> float:
        a = lambda_t / self.mu
        if a <= 0:
            return 0.0
        return self.erlang_b(a)

    def compute_p_block_nonstationary(
        self, lambda_series: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        result = []
        for t, lam in lambda_series:
            rho = self.compute_utilization(lam)
            a = lam / self.mu
            if rho >= 1:
                p0_inv = sum((a ** n) / factorial(n) for n in range(self.c))
                p0_inv += ((a ** self.c) / factorial(self.c)) * (self.K - self.c + 1)
                p0 = 1 / p0_inv if p0_inv > 0 else 0
                p_block = p0 * ((a ** self.c) / factorial(self.c))
                p_block = min(p_block, 1.0)
            else:
                p_block = self.erlang_b(a) * 0.1
            result.append((t, p_block))
            if p_block > 0.5:
                self._blocking_events.append(
                    BlockingEvent(
                        timestamp=t,
                        queue_length=self.K,
                        arrival_rate=lam,
                        utilization=rho,
                    )
                )
        return result

    def compute_d_loss(
        self,
        p_block_series: List[Tuple[float, float]],
        n_series: List[Tuple[float, float]],
        dt: float,
    ) -> DLossResult:
        if not p_block_series or not n_series:
            return DLossResult(
                d_loss=0.0,
                numerator=0.0,
                denominator=0.0,
                dt=dt,
                sla_compliant=True,
                violation_magnitude=None,
            )
        p_dict = {t: p for t, p in p_block_series}
        numerator = 0.0
        denominator = 0.0
        for t, n in n_series:
            p = p_dict.get(t, 0.0)
            numerator += p * n * dt
            denominator += n * dt
        if denominator == 0:
            d_loss = 0.0
        else:
            d_loss = numerator / denominator
        sla_compliant = d_loss <= self.config.sla_delta
        violation_mag = None
        if not sla_compliant:
            violation_mag = (d_loss - self.config.sla_delta) / self.config.sla_delta * 100
        return DLossResult(
            d_loss=d_loss,
            numerator=numerator,
            denominator=denominator,
            dt=dt,
            sla_compliant=sla_compliant,
            violation_magnitude=violation_mag,
        )


    def compute_e_t_fail(
        self, rho_series: List[Tuple[float, float]], rho_critical: float = 0.95
    ) -> float:
        if not rho_series or len(rho_series) < 2:
            return float("inf")
        last_rho = rho_series[-1][1]
        if last_rho >= rho_critical:
            return 0.0
        rho_values = [r for _, r in rho_series]
        if len(rho_values) >= 2:
            trend = rho_values[-1] - rho_values[-2]
            if trend <= 0:
                return float("inf")
        margin = rho_critical - last_rho
        lambda_margin = margin * self.c * self.mu
        if lambda_margin <= 0:
            return float("inf")
        return 1 / lambda_margin

    def analyze(self, traffic: TrafficTimeSeries) -> QueueingAnalysisResult:
        lambda_series = list(zip(traffic.timestamps, traffic.n_total))
        utilization = self.compute_utilization_series(lambda_series)
        p_block_series = self.compute_p_block_nonstationary(lambda_series)
        n_series = list(zip(traffic.timestamps, traffic.n_total))
        dt = traffic.metadata.get("time_step", 1.0)
        d_loss = self.compute_d_loss(p_block_series, n_series, dt)
        rho_series = list(zip(utilization.timestamps, utilization.instantaneous))
        e_t_fail = self.compute_e_t_fail(rho_series)
        metrics_timeline = []
        for i, t in enumerate(traffic.timestamps):
            rho = utilization.instantaneous[i] if i < len(utilization.instantaneous) else 0
            p_b = p_block_series[i][1] if i < len(p_block_series) else 0
            throughput = traffic.n_total[i] * (1 - p_b)
            metrics_timeline.append(
                QueueingMetricsSnapshot(
                    timestamp=t,
                    rho=rho,
                    p_block=p_b,
                    queue_length=min(rho * self.c, self.K),
                    throughput=throughput,
                    e_wait=rho / (1 - rho) / self.mu if rho < 1 else float("inf"),
                )
            )
        return QueueingAnalysisResult(
            config=self.config,
            utilization=utilization,
            p_block_series=p_block_series,
            d_loss=d_loss,
            e_t_fail=e_t_fail,
            blocking_events=self._blocking_events,
            metrics_timeline=metrics_timeline,
        )


    def generate_report(self, analysis: QueueingAnalysisResult) -> SystemReport:
        sla_compliance = []
        violations = []
        p_block_avg = (
            sum(p for _, p in analysis.p_block_series) / len(analysis.p_block_series)
            if analysis.p_block_series
            else 0
        )
        p_block_status = SLAComplianceStatus.COMPLIANT
        p_block_severity = None
        if p_block_avg > self.config.sla_epsilon:
            p_block_status = SLAComplianceStatus.VIOLATION
            p_block_severity = (p_block_avg - self.config.sla_epsilon) / self.config.sla_epsilon * 100
            violations.append(
                SLAViolation(
                    metric_name="P_block",
                    actual_value=p_block_avg,
                    threshold=self.config.sla_epsilon,
                    severity_percent=p_block_severity,
                    start_time=analysis.utilization.timestamps[0] if analysis.utilization.timestamps else 0,
                    duration=analysis.utilization.timestamps[-1] - analysis.utilization.timestamps[0]
                    if len(analysis.utilization.timestamps) > 1
                    else 0,
                )
            )
        sla_compliance.append(
            SLAComplianceResult(
                metric_name="P_block",
                actual_value=p_block_avg,
                threshold=self.config.sla_epsilon,
                status=p_block_status,
                violation_severity_percent=p_block_severity,
            )
        )
        d_loss_status = SLAComplianceStatus.COMPLIANT
        d_loss_severity = None
        if not analysis.d_loss.sla_compliant:
            d_loss_status = SLAComplianceStatus.VIOLATION
            d_loss_severity = analysis.d_loss.violation_magnitude
            violations.append(
                SLAViolation(
                    metric_name="D_loss",
                    actual_value=analysis.d_loss.d_loss,
                    threshold=self.config.sla_delta,
                    severity_percent=d_loss_severity or 0,
                    start_time=analysis.utilization.timestamps[0] if analysis.utilization.timestamps else 0,
                    duration=analysis.utilization.timestamps[-1] - analysis.utilization.timestamps[0]
                    if len(analysis.utilization.timestamps) > 1
                    else 0,
                )
            )
        sla_compliance.append(
            SLAComplianceResult(
                metric_name="D_loss",
                actual_value=analysis.d_loss.d_loss,
                threshold=self.config.sla_delta,
                status=d_loss_status,
                violation_severity_percent=d_loss_severity,
            )
        )
        recommendations = []
        if violations:
            recommendations.append("Consider increasing server count (c) or service rate (mu)")
            recommendations.append("Review queue capacity (K) for potential increase")
        avg_throughput = (
            sum(m.throughput for m in analysis.metrics_timeline) / len(analysis.metrics_timeline)
            if analysis.metrics_timeline
            else 0
        )
        avg_queue = (
            sum(m.queue_length for m in analysis.metrics_timeline) / len(analysis.metrics_timeline)
            if analysis.metrics_timeline
            else 0
        )
        return SystemReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            simulation_params={"c": self.c, "K": self.K, "mu": self.mu},
            utilization_summary={
                "max": analysis.utilization.max_utilization,
                "avg": analysis.utilization.avg_utilization,
            },
            p_block_summary={"avg": p_block_avg, "max": max(p for _, p in analysis.p_block_series) if analysis.p_block_series else 0},
            d_loss=analysis.d_loss.d_loss,
            e_t_fail=analysis.e_t_fail,
            throughput=avg_throughput,
            avg_queue_length=avg_queue,
            sla_compliance=sla_compliance,
            violations=violations,
            overall_compliant=len(violations) == 0,
            recommendations=recommendations,
        )


    def compare_analytical_vs_simulated(
        self,
        analytical: QueueingAnalysisResult,
        simulated: QueueingAnalysisResult,
    ) -> ComparisonReport:
        analytical_metrics = {
            "rho_avg": analytical.utilization.avg_utilization,
            "rho_max": analytical.utilization.max_utilization,
            "d_loss": analytical.d_loss.d_loss,
            "e_t_fail": analytical.e_t_fail if analytical.e_t_fail != float("inf") else 1e10,
        }
        simulated_metrics = {
            "rho_avg": simulated.utilization.avg_utilization,
            "rho_max": simulated.utilization.max_utilization,
            "d_loss": simulated.d_loss.d_loss,
            "e_t_fail": simulated.e_t_fail if simulated.e_t_fail != float("inf") else 1e10,
        }
        relative_errors = {}
        discrepancies = []
        for key in analytical_metrics:
            a_val = analytical_metrics[key]
            s_val = simulated_metrics[key]
            if a_val != 0:
                error = abs(a_val - s_val) / abs(a_val) * 100
            else:
                error = 0 if s_val == 0 else 100
            relative_errors[key] = error
            if error > 10:
                discrepancies.append(f"{key}: relative error {error:.2f}% exceeds 10% threshold")
        sample_size = len(analytical.metrics_timeline)
        confidence_intervals = {}
        for key in analytical_metrics:
            val = analytical_metrics[key]
            margin = val * 0.05
            confidence_intervals[key] = (val - margin, val + margin)
        return ComparisonReport(
            analytical_metrics=analytical_metrics,
            simulated_metrics=simulated_metrics,
            relative_errors=relative_errors,
            confidence_intervals=confidence_intervals,
            discrepancies=discrepancies,
            sample_size=sample_size,
        )
