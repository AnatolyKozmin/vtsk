#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "sbp_100k.jsonl"
ARTIFACTS_DIR = ROOT / "training" / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_no}: {e}") from e
    return rows


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    records = []

    for row in rows:
        data = row.get("Data", {})
        meta = row.get("Meta", {})

        ts_raw = safe_get(row, "Data", "CurrentTimestamp")
        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")

        amount_str = safe_get(row, "Data", "Amount", default="0")
        try:
            amount = float(amount_str)
        except (TypeError, ValueError):
            amount = 0.0

        payer_client_id = safe_get(row, "Data", "PayerData", "ClientId", default="")
        payer_bic = safe_get(row, "Data", "PayerData", "PayerBIC", default="")
        beneficiary_bic = safe_get(row, "Data", "BeneficiaryData", "BeneficiaryBIC", default="")
        trn_type = safe_get(row, "Data", "TrnType", default="")
        narrative = safe_get(row, "Data", "Narrative", default="")
        trn_id = safe_get(row, "Data", "TrnId", default="")

        base_label = safe_get(row, "Meta", "base_label", default=0)
        velocity_anomaly = bool(safe_get(row, "Meta", "velocity_anomaly", default=False))
        is_anomaly = bool(safe_get(row, "Meta", "is_anomaly", default=False))

        try:
            base_label = int(base_label)
        except (TypeError, ValueError):
            base_label = 0

        record = {
            "timestamp": ts,
            "trn_id": trn_id,
            "trn_type": trn_type,
            "payer_client_id": payer_client_id,
            "payer_bic": payer_bic,
            "beneficiary_bic": beneficiary_bic,
            "amount": amount,
            "currency": safe_get(row, "Data", "Currency", default=""),
            "narrative": narrative,
            "base_label": base_label,
            "is_anomaly_meta": is_anomaly,
            "velocity_anomaly": velocity_anomaly,
        }
        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("Dataset is empty after parsing.")

    # Basic cleaning
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Base features
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["amount"] = df["amount"].clip(lower=0.0)
    df["log_amount"] = df["amount"].apply(lambda x: math.log1p(x))

    df["narrative_len"] = df["narrative"].fillna("").str.len()

    # Targets
    df["target_binary"] = (df["base_label"] != 0).astype(int)
    df["target_multiclass"] = df["base_label"].astype(int)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Первые простые признаки по истории.
    Делаем их без утечки: только по прошлым транзакциям.
    """
    df = df.copy()

    # время с предыдущей транзакции клиента
    df["prev_ts_client"] = df.groupby("payer_client_id")["timestamp"].shift(1)
    delta = (df["timestamp"] - df["prev_ts_client"]).dt.total_seconds()
    df["secs_since_prev_client_tx"] = delta.fillna(-1)

    # предыдущая сумма клиента
    df["prev_amount_client"] = df.groupby("payer_client_id")["amount"].shift(1)
    df["prev_amount_client"] = df["prev_amount_client"].fillna(0.0)

    # накопительная средняя сумма клиента по прошлому
    grp = df.groupby("payer_client_id")["amount"]
    cum_sum = grp.cumsum() - df["amount"]
    cum_cnt = grp.cumcount()
    df["client_amount_mean_past"] = (cum_sum / cum_cnt.replace(0, pd.NA)).fillna(df["amount"].median())

    # отклонение от прошлой средней клиента
    df["amount_to_client_mean_ratio"] = (
        df["amount"] / df["client_amount_mean_past"].replace(0, pd.NA)
    ).fillna(1.0)

    # новая ли пара client -> beneficiary_bic
    seen_pairs = set()
    new_pair_flags = []
    for _, row in df.iterrows():
        pair = (row["payer_client_id"], row["beneficiary_bic"])
        if pair in seen_pairs:
            new_pair_flags.append(0)
        else:
            new_pair_flags.append(1)
            seen_pairs.add(pair)
    df["is_new_pair"] = new_pair_flags

    # простые rolling-признаки по клиенту: число транзакций за 5 минут
    # делаем через проход, чтобы не было боли с groupby-rolling на старте
    from collections import defaultdict, deque

    queues = defaultdict(deque)
    tx_count_5m = []

    for _, row in df.iterrows():
        cid = row["payer_client_id"]
        ts = row["timestamp"].to_pydatetime()
        q = queues[cid]

        cutoff = ts - pd.Timedelta(minutes=5).to_pytimedelta()
        while q and q[0] < cutoff:
            q.popleft()

        tx_count_5m.append(len(q))
        q.append(ts)

    df["client_tx_count_prev_5m"] = tx_count_5m

    # глобальная активность за 5 минут
    global_q = deque()
    global_count_5m = []

    for _, row in df.iterrows():
        ts = row["timestamp"].to_pydatetime()
        cutoff = ts - pd.Timedelta(minutes=5).to_pytimedelta()

        while global_q and global_q[0] < cutoff:
            global_q.popleft()

        global_count_5m.append(len(global_q))
        global_q.append(ts)

    df["global_tx_count_prev_5m"] = global_count_5m

    # cleanup
    df = df.drop(columns=["prev_ts_client"])

    return df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Делим по времени:
    70% train, 15% val, 15% test
    """
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def save_outputs(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    full_csv = ARTIFACTS_DIR / "dataset_full.csv"
    train_csv = ARTIFACTS_DIR / "train.csv"
    val_csv = ARTIFACTS_DIR / "val.csv"
    test_csv = ARTIFACTS_DIR / "test.csv"

    full_df.to_csv(full_csv, index=False, encoding="utf-8")
    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")

    # parquet пробуем сохранить, если есть движок
    try:
        full_df.to_parquet(ARTIFACTS_DIR / "dataset_full.parquet", index=False)
        train_df.to_parquet(ARTIFACTS_DIR / "train.parquet", index=False)
        val_df.to_parquet(ARTIFACTS_DIR / "val.parquet", index=False)
        test_df.to_parquet(ARTIFACTS_DIR / "test.parquet", index=False)
        parquet_msg = "CSV + Parquet"
    except Exception:
        parquet_msg = "CSV only"

    print(f"Saved artifacts: {parquet_msg}")
    print(f"- {full_csv}")
    print(f"- {train_csv}")
    print(f"- {val_csv}")
    print(f"- {test_csv}")


def print_summary(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== DATASET SUMMARY ===")
    print(f"Total rows: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    print("\nBase label distribution:")
    print(df["base_label"].value_counts().sort_index())

    print("\nBinary target distribution:")
    print(df["target_binary"].value_counts().sort_index())

    print("\nAmount stats by label:")
    stats = df.groupby("base_label")["amount"].agg(["count", "mean", "median", "max"])
    print(stats)

    print("\nSample columns:")
    print(df.columns.tolist())


def main():
    print(f"Loading: {INPUT_PATH}")
    rows = load_jsonl(INPUT_PATH)

    print("Building dataframe...")
    df = build_dataframe(rows)

    print("Adding temporal features...")
    df = add_temporal_features(df)

    print("Splitting by time...")
    train_df, val_df, test_df = split_by_time(df)

    print("Saving artifacts...")
    save_outputs(df, train_df, val_df, test_df)

    print_summary(df, train_df, val_df, test_df)


if __name__ == "__main__":
    main()