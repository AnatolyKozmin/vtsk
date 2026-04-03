"""
Lightweight asyncpg connection pool for model-api.

Writes ML block events to the shared PostgreSQL instance
(same tables the receiver uses, so the monitor can query everything
from one place).

Tables used:
  traffic_responses  — one row per ML-blocked request
  protection_events  — one row per ML block with full anomaly details
  blocked_requests   — one row per ML-blocked request (for the WAF-like feed)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg

logger = logging.getLogger("model-api.db")

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> None:
    global _pool
    dsn = (
        f"postgresql://{os.getenv('DB_USER', 'vtsk')}"
        f":{os.getenv('DB_PASSWORD', '1234')}"
        f"@{os.getenv('DB_HOST', 'postgres')}"
        f":{os.getenv('DB_PORT', '5432')}"
        f"/{os.getenv('DB_NAME', 'vtsk_db')}"
    )
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    logger.info("DB pool ready")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def record_ml_block(
    *,
    request_id: str,
    batch_id: str,
    session_id: str,
    sender: str,
    amount: float,
    anomaly_prob: float,
    anomaly_type: Optional[int],
    anomaly_label: Optional[str],
) -> None:
    """Persist one ML-blocked request to the database."""
    if _pool is None:
        return

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # store as naive UTC
    details = json.dumps({
        "sender": sender,
        "amount": amount,
        "anomaly_label": anomaly_label,
        "prob": round(anomaly_prob, 4),
        "anomaly_type": anomaly_type,
    })

    try:
        async with _pool.acquire() as conn:
            # --- traffic_responses ---
            await conn.execute(
                """
                INSERT INTO traffic_responses
                    (request_id, batch_id, received_at, response_time_ms,
                     status_code, was_blocked, blocked_by, passed_through)
                VALUES ($1, $2, $3, 0, 403, TRUE, 'ml_proxy', FALSE)
                ON CONFLICT DO NOTHING
                """,
                request_id, batch_id, now,
            )

            # --- protection_events ---
            await conn.execute(
                """
                INSERT INTO protection_events
                    (session_id, event_type, source, timestamp,
                     details, severity, action_taken)
                VALUES ($1, 'block', 'ml_proxy', $2, $3, 'high', 'blocked')
                """,
                session_id, now, details,
            )

            # --- blocked_requests ---
            await conn.execute(
                """
                INSERT INTO blocked_requests
                    (request_id, session_id, blocked_at, blocked_by,
                     block_reason, attack_signature)
                VALUES ($1, $2, $3, 'ml_proxy', $4, $5)
                ON CONFLICT DO NOTHING
                """,
                request_id, session_id, now,
                f"{anomaly_label} (p={anomaly_prob:.3f})",
                anomaly_label or "unknown",
            )
    except Exception as exc:
        logger.warning("DB write failed for %s: %s", request_id, exc)
