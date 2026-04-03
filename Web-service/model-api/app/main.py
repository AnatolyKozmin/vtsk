"""
Model-API — behavioural ML inspection proxy.

Sits between pfsense-a and pfsense-b in transit_net:

  sender:5000 → pfsense-a:5001 → [model-api:5001] → pfsense-b:5001 → receiver:5001

For every POST /receive the proxy:
  1. Parses the batch: { batch_id, session_id, requests: [...] }
  2. For each request, extracts features and scores with CatBoost binary model.
  3. Requests with anomaly probability ≥ BINARY_THRESHOLD are BLOCKED:
       - removed from the forwarded batch
       - classified by the multiclass model (anomaly subtype)
       - logged as "ml_proxy" block events
  4. The cleaned batch (only non-anomalous requests) is forwarded downstream.
  5. Response merges downstream results with local block results.

What CatBoost detects here (behavioural layer):
  - Velocity spikes: a sender firing many requests within 5 minutes
  - Amount anomalies: amounts deviating from a sender's historical baseline
  - New sender→receiver pairs with unusual amounts
  - Other patterns learned from SBP training data mapped to HTTP traffic

Signature-based detection (SQLi, XSS, path traversal, etc.) is handled
by the receiver — the two layers are complementary.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .db import close_pool, init_pool, record_ml_block
from .features import build_feature_row, parse_request
from .state import StateStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
DOWNSTREAM_URL = os.getenv("DOWNSTREAM_URL", "http://pfsense-b:5001").rstrip("/")
BINARY_THRESHOLD = float(os.getenv("BINARY_THRESHOLD", "0.5"))

ANOMALY_TYPE_LABELS = {
    1: "velocity_spike",
    2: "amount_spike_exponential",
    3: "amount_spike_poisson",
    4: "amount_spike_pareto",
}

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

binary_model: Optional[CatBoostClassifier] = None
multiclass_model: Optional[CatBoostClassifier] = None
store: StateStore = StateStore()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("model-api")

_KNOWN_CATEGORICALS = {"payer_client_id", "payer_bic", "beneficiary_bic", "trn_type", "narrative"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global binary_model, multiclass_model

    binary_path = MODELS_DIR / "binary_model.cbm"
    multiclass_path = MODELS_DIR / "multiclass_model.cbm"

    if not binary_path.exists():
        raise FileNotFoundError(f"Binary model not found: {binary_path}")
    if not multiclass_path.exists():
        raise FileNotFoundError(f"Multiclass model not found: {multiclass_path}")

    binary_model = CatBoostClassifier()
    binary_model.load_model(str(binary_path))

    multiclass_model = CatBoostClassifier()
    multiclass_model.load_model(str(multiclass_path))

    await init_pool()
    logger.info("Models loaded. Downstream: %s | Threshold: %.2f", DOWNSTREAM_URL, BINARY_THRESHOLD)
    yield
    await close_pool()


app = FastAPI(title="Model-API Proxy", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dataframe(row: dict, model: CatBoostClassifier) -> pd.DataFrame:
    record = {}
    for col in model.feature_names_:
        val = row.get(col)
        if col in _KNOWN_CATEGORICALS:
            record[col] = str(val) if val is not None else "NA"
        else:
            record[col] = float(val) if val is not None else 0.0
    return pd.DataFrame([record])


async def _score_request(req: dict) -> dict:
    """
    Score one request dict.  Returns:
        blocked       bool
        anomaly_prob  float
        anomaly_type  int|None
        anomaly_label str|None
        request_id    str
    """
    static = parse_request(req)

    stateful = await store.compute_and_update(
        client_id=static["payer_client_id"],
        beneficiary_bic=static["beneficiary_bic"],
        amount=static["amount"],
        ts=static["timestamp"],
    )

    features = build_feature_row(static, stateful)

    X_bin = _to_dataframe(features, binary_model)
    prob: float = float(binary_model.predict_proba(X_bin)[0, 1])
    blocked = prob >= BINARY_THRESHOLD

    anomaly_type: Optional[int] = None
    anomaly_label: Optional[str] = None

    if blocked:
        X_mc = _to_dataframe(features, multiclass_model)
        mc_pred = int(multiclass_model.predict(X_mc)[0])
        anomaly_type = mc_pred
        anomaly_label = ANOMALY_TYPE_LABELS.get(mc_pred, f"type_{mc_pred}")

    return {
        "blocked": blocked,
        "anomaly_prob": round(prob, 4),
        "anomaly_type": anomaly_type,
        "anomaly_label": anomaly_label,
        "request_id": static["request_id"],
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": binary_model is not None and multiclass_model is not None,
        "downstream": DOWNSTREAM_URL,
        "threshold": BINARY_THRESHOLD,
    }


@app.post("/receive")
async def receive(request: Request) -> Response:
    """
    Main inspection endpoint — mirrors receiver's /receive signature.
    Filters the batch, forwards the clean subset, returns merged results.
    """
    raw_body = await request.body()

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    batch_id = body.get("batch_id", "unknown")
    session_id = body.get("session_id", "unknown")
    requests_list = body.get("requests", [])
    if not isinstance(requests_list, list):
        requests_list = [requests_list]

    clean_requests = []
    blocked_results = []

    for req in requests_list:
        try:
            decision = await _score_request(req)
        except (ValueError, KeyError) as exc:
            logger.debug("Cannot score request %s: %s", req.get("request_id"), exc)
            clean_requests.append(req)
            continue

        if decision["blocked"]:
            sender = req.get("payload", {}).get("sender", "?")
            amount = float(req.get("payload", {}).get("amount", 0) or 0)
            logger.info(
                "[BLOCKED] id=%s sender=%s amount=%.0f prob=%.4f type=%s",
                decision["request_id"], sender, amount,
                decision["anomaly_prob"], decision["anomaly_label"] or "-",
            )
            blocked_results.append({
                "request_id": decision["request_id"],
                "received": True,
                "response_time_ms": 0,
                "was_blocked": True,
                "blocked_by": "ml_proxy",
                "status_code": 403,
                "anomaly_probability": decision["anomaly_prob"],
                "anomaly_label": decision["anomaly_label"],
            })
            # persist to DB asynchronously (fire-and-forget, don't block proxy)
            import asyncio
            asyncio.create_task(record_ml_block(
                request_id=decision["request_id"],
                batch_id=batch_id,
                session_id=session_id,
                sender=sender,
                amount=amount,
                anomaly_prob=decision["anomaly_prob"],
                anomaly_type=decision["anomaly_type"],
                anomaly_label=decision["anomaly_label"],
            ))
        else:
            clean_requests.append(req)

    # Forward the filtered batch downstream (may be empty)
    downstream_results = []
    if clean_requests:
        forward_body = {
            "batch_id": batch_id,
            "session_id": session_id,
            "requests": clean_requests,
        }
        skip_headers = {"host", "content-length", "transfer-encoding", "connection"}
        forward_headers = {
            k: v for k, v in request.headers.items() if k.lower() not in skip_headers
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{DOWNSTREAM_URL}/receive",
                    content=json.dumps(forward_body),
                    headers={**forward_headers, "Content-Type": "application/json"},
                )
            downstream_body = resp.json()
            downstream_results = downstream_body.get("results", [])
        except httpx.RequestError as exc:
            logger.error("Downstream unreachable: %s", exc)
            # Treat all clean requests as errors
            for req in clean_requests:
                downstream_results.append({
                    "request_id": req.get("request_id", "unknown"),
                    "received": False,
                    "error": f"downstream unreachable: {exc}",
                    "was_blocked": False,
                    "status_code": 502,
                })

    all_results = downstream_results + blocked_results
    total = len(all_results)
    blocked_count = len(blocked_results)
    # Count blocks done by downstream too
    total_blocked = sum(1 for r in all_results if r.get("was_blocked"))

    return JSONResponse(
        status_code=200,
        content={
            "batch_id": batch_id,
            "session_id": session_id,
            "total_requests": total,
            "received_count": total,
            "blocked_count": total_blocked,
            "ml_blocked_count": blocked_count,
            "results": all_results,
        },
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_other(request: Request, path: str) -> Response:
    """Transparent proxy for all non-/receive endpoints (health, stats, etc.)."""
    body = await request.body()
    skip_headers = {"host", "content-length", "transfer-encoding", "connection"}
    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in skip_headers
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method=request.method,
                url=f"{DOWNSTREAM_URL}/{path}",
                headers=forward_headers,
                content=body,
                params=dict(request.query_params),
            )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "downstream unreachable", "detail": str(exc)},
        )

    relay_headers = {k: v for k, v in resp.headers.items() if k.lower() not in skip_headers}
    return Response(content=resp.content, status_code=resp.status_code, headers=relay_headers)
