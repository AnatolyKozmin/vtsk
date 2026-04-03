"""
Feature engineering: maps a single web-service request object
onto the feature space the CatBoost models were trained on.

Web-service request format (one element of the "requests" array):
{
    "request_id": "REQ-...",
    "timestamp":  "2024-01-01T12:00:00+00:00",
    "payload": {
        "transaction_id": "...",
        "amount":          1500.75,
        "currency":        "RUB",
        "sender":          "user_42",
        "receiver":        "user_99",
        "description":     "...",
        "timestamp":       "...",
        # malicious extras may be present:
        "id":              "' OR '1'='1",
        "_attack_type":    "sql_injection",
        "_pattern":        "...",
    },
    "is_malicious":  true/false,     # sent by generator, NOT visible to the model
    "attack_type":   "sql_injection",# same - not used for scoring
    "headers":       {...}
}

Mapping to CatBoost feature space:
  sender    → payer_client_id  (client identity)
  receiver  → beneficiary_bic  (destination identity)
  "HTTP"    → payer_bic        (constant — all requests share the same channel)
  "HTTP"    → trn_type         (constant — all are HTTP transactions)
  amount    → amount / log_amount
  timestamp → hour, minute, day_of_week, is_weekend

Stateful features (velocity, rolling means, new-pair flag) are computed
by StateStore and merged in separately.
"""

import math
from datetime import datetime, timezone
from typing import Any


def _safe(d: dict, *keys, default=None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_request(req: dict) -> dict:
    """
    Parse one item from the 'requests' array into typed fields + static features.
    Raises ValueError if the request doesn't have the expected shape.
    """
    payload = req.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Missing or invalid 'payload' in request object")

    # --- timestamp ---
    ts_raw = req.get("timestamp") or payload.get("timestamp")
    try:
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception as exc:
        raise ValueError(f"Cannot parse timestamp: {ts_raw!r}") from exc

    # --- amount ---
    try:
        amount = max(float(payload.get("amount", 0) or 0), 0.0)
    except (TypeError, ValueError):
        amount = 0.0

    # --- identity fields ---
    # sender maps to payer_client_id (driving per-client state)
    client_id = str(payload.get("sender") or "unknown")
    # receiver maps to beneficiary_bic (destination tracking)
    beneficiary = str(payload.get("receiver") or "unknown")
    # constant channel identifier — all traffic arrives via the same HTTP channel
    payer_bic = "HTTP_CHANNEL"
    trn_type = str(payload.get("currency") or "HTTP")

    narrative = str(payload.get("description") or "")

    return {
        # --- identifiers (not model features, for logging only) ---
        "request_id": str(req.get("request_id") or ""),
        "timestamp": ts,
        # --- static model features ---
        "amount": amount,
        "log_amount": math.log1p(amount),
        "hour": ts.hour,
        "minute": ts.minute,
        "day_of_week": ts.weekday(),          # 0=Mon … 6=Sun
        "is_weekend": int(ts.weekday() >= 5),
        "narrative_len": len(narrative),
        # --- categorical model features ---
        "payer_client_id": client_id,
        "payer_bic": payer_bic,
        "beneficiary_bic": beneficiary,
        "trn_type": trn_type,
    }


def build_feature_row(static: dict, stateful: dict) -> dict:
    """Merge static and stateful feature dicts into one flat row."""
    return {**static, **stateful}
