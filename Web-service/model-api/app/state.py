"""
In-memory stateful store for per-client and global transaction features.

All state is protected by a single asyncio.Lock so the FastAPI event loop
(single-threaded) can safely share it across concurrent requests.
"""

import asyncio
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


WINDOW_SECONDS = 300  # 5-minute rolling window for velocity features


@dataclass
class _ClientState:
    last_ts: Optional[datetime] = None
    last_amount: float = 0.0
    amount_sum: float = 0.0
    amount_count: int = 0
    # timestamps (epoch seconds) of recent transactions for the 5-min window
    recent_ts: deque = field(default_factory=deque)
    # beneficiary BICs this client has already sent to
    seen_beneficiaries: set = field(default_factory=set)


class StateStore:
    """
    Thread-safe (asyncio) feature store.

    Call `compute_and_update(...)` once per incoming transaction.
    It returns the stateful features computed from *past* history,
    then updates the history so the next call sees the current tx.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._clients: dict[str, _ClientState] = defaultdict(_ClientState)
        self._global_ts: deque = deque()  # epoch seconds of all recent txns

    async def compute_and_update(
        self,
        client_id: str,
        beneficiary_bic: str,
        amount: float,
        ts: datetime,
    ) -> dict:
        """
        Returns a dict with stateful features for the given transaction,
        computed from history *before* this transaction is recorded.
        After returning, history is updated to include this transaction.
        """
        async with self._lock:
            state = self._clients[client_id]
            now_epoch = ts.timestamp()
            cutoff = now_epoch - WINDOW_SECONDS

            # --- secs_since_prev_client_tx ---
            if state.last_ts is not None:
                secs_since = (ts - state.last_ts).total_seconds()
            else:
                secs_since = -1.0

            # --- prev_amount_client ---
            prev_amount = state.last_amount

            # --- client_amount_mean_past ---
            if state.amount_count > 0:
                client_mean = state.amount_sum / state.amount_count
            else:
                # First transaction from this client — use own amount as baseline
                client_mean = amount if amount > 0 else 1.0

            # --- amount_to_client_mean_ratio ---
            ratio = amount / client_mean if client_mean > 0 else 1.0

            # --- is_new_pair ---
            is_new_pair = 1 if beneficiary_bic not in state.seen_beneficiaries else 0

            # --- client_tx_count_prev_5m (before this tx) ---
            while state.recent_ts and state.recent_ts[0] < cutoff:
                state.recent_ts.popleft()
            client_5m = len(state.recent_ts)

            # --- global_tx_count_prev_5m (before this tx) ---
            while self._global_ts and self._global_ts[0] < cutoff:
                self._global_ts.popleft()
            global_5m = len(self._global_ts)

            # ---- Update state (must happen AFTER feature extraction) ----
            state.last_ts = ts
            state.last_amount = amount
            state.amount_sum += amount
            state.amount_count += 1
            state.recent_ts.append(now_epoch)
            self._global_ts.append(now_epoch)
            state.seen_beneficiaries.add(beneficiary_bic)

        return {
            "secs_since_prev_client_tx": secs_since,
            "prev_amount_client": prev_amount,
            "client_amount_mean_past": client_mean,
            "amount_to_client_mean_ratio": ratio,
            "is_new_pair": is_new_pair,
            "client_tx_count_prev_5m": client_5m,
            "global_tx_count_prev_5m": global_5m,
        }
