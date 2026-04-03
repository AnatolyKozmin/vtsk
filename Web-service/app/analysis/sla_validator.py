from typing import List
from app.models.queueing import SLAComplianceResult, SLAComplianceStatus, SLAViolation


class SLAValidator:
    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta

    def validate_p_block(self, p_block: float) -> SLAComplianceResult:
        if p_block <= self.epsilon:
            return SLAComplianceResult(
                metric_name="P_block",
                actual_value=p_block,
                threshold=self.epsilon,
                status=SLAComplianceStatus.COMPLIANT,
                violation_severity_percent=None,
            )
        severity = self.calculate_violation_severity(p_block, self.epsilon)
        return SLAComplianceResult(
            metric_name="P_block",
            actual_value=p_block,
            threshold=self.epsilon,
            status=SLAComplianceStatus.VIOLATION,
            violation_severity_percent=severity,
        )

    def validate_d_loss(self, d_loss: float) -> SLAComplianceResult:
        if d_loss <= self.delta:
            return SLAComplianceResult(
                metric_name="D_loss",
                actual_value=d_loss,
                threshold=self.delta,
                status=SLAComplianceStatus.COMPLIANT,
                violation_severity_percent=None,
            )
        severity = self.calculate_violation_severity(d_loss, self.delta)
        return SLAComplianceResult(
            metric_name="D_loss",
            actual_value=d_loss,
            threshold=self.delta,
            status=SLAComplianceStatus.VIOLATION,
            violation_severity_percent=severity,
        )

    def calculate_violation_severity(self, actual: float, threshold: float) -> float:
        if threshold == 0:
            return 100.0 if actual > 0 else 0.0
        return (actual - threshold) / threshold * 100

    def generate_recommendations(self, violations: List[SLAViolation]) -> List[str]:
        recommendations = []
        for v in violations:
            if v.metric_name == "P_block":
                recommendations.append(
                    f"P_block violation ({v.severity_percent:.1f}%): increase server count or service rate"
                )
            elif v.metric_name == "D_loss":
                recommendations.append(
                    f"D_loss violation ({v.severity_percent:.1f}%): increase queue capacity or reduce load"
                )
        if not recommendations:
            recommendations.append("All SLA metrics are within acceptable thresholds")
        return recommendations
