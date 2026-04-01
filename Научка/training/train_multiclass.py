#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "training" / "artifacts"
MODELS_DIR = ROOT / "training" / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

FULL_DATASET_PATH = ARTIFACTS_DIR / "dataset_full.csv"


def load_data():
    df = pd.read_csv(FULL_DATASET_PATH)
    return df


def prepare_multiclass_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["target_multiclass"] != 0].copy()
    df["target_multiclass"] = df["target_multiclass"].astype(int)
    return df


def make_splits(df: pd.DataFrame):
    # 70 / 15 / 15, стратифицированно по классам 1..4
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["target_multiclass"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["target_multiclass"],
    )

    return train_df, val_df, test_df


def select_features(df: pd.DataFrame):
    feature_cols = [
        "amount",
        "log_amount",
        "payer_client_id",
        "payer_bic",
        "beneficiary_bic",
        "trn_type",
        "secs_since_prev_client_tx",
        "prev_amount_client",
        "client_amount_mean_past",
        "amount_to_client_mean_ratio",
        "is_new_pair",
        "client_tx_count_prev_5m",
        "global_tx_count_prev_5m",
    ]

    cat_features = [
        "payer_client_id",
        "payer_bic",
        "beneficiary_bic",
        "trn_type",
    ]

    X = df[feature_cols].copy()
    y = df["target_multiclass"].astype(int).copy()

    for col in cat_features:
        X[col] = X[col].astype(str)

    return X, y, feature_cols, cat_features


def main():
    print("Loading full dataset...")
    df = load_data()
    df = prepare_multiclass_df(df)

    print("Creating stratified splits...")
    train_df, val_df, test_df = make_splits(df)

    X_train, y_train, feature_cols, cat_features = select_features(train_df)
    X_val, y_val, _, _ = select_features(val_df)
    X_test, y_test, _, _ = select_features(test_df)

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    print("\nTrain multiclass distribution:")
    print(y_train.value_counts().sort_index())

    print("\nVal multiclass distribution:")
    print(y_val.value_counts().sort_index())

    print("\nTest multiclass distribution:")
    print(y_test.value_counts().sort_index())

    model = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        verbose=100,
    )

    print("\nTraining multiclass model...")
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test).astype(int).ravel()

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    model_path = MODELS_DIR / "multiclass_model.cbm"
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    print("\n=== FEATURE IMPORTANCE (top 15) ===")
    importances = model.get_feature_importance()
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(fi.head(15).to_string(index=False))

    fi_path = ARTIFACTS_DIR / "multiclass_feature_importance.csv"
    fi.to_csv(fi_path, index=False, encoding="utf-8")
    print(f"\nFeature importance saved to: {fi_path}")


if __name__ == "__main__":
    main()