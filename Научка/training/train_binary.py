#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "training" / "artifacts"
MODELS_DIR = ROOT / "training" / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ARTIFACTS_DIR / "train.csv"
VAL_PATH = ARTIFACTS_DIR / "val.csv"
TEST_PATH = ARTIFACTS_DIR / "test.csv"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df


def select_features(df: pd.DataFrame):
    feature_cols = [
        "amount",
        "log_amount",
        "day_of_week",
        "is_weekend",
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
    y = df["target_binary"].astype(int).copy()

    # CatBoost сам умеет NaN, но приведём строки к str на всякий случай
    for col in cat_features:
        X[col] = X[col].astype(str)

    return X, y, feature_cols, cat_features


def main():
    print("Loading datasets...")
    train_df, val_df, test_df = load_data()

    X_train, y_train, feature_cols, cat_features = select_features(train_df)
    X_val, y_val, _, _ = select_features(val_df)
    X_test, y_test, _, _ = select_features(test_df)

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    print("\nTrain binary target distribution:")
    print(y_train.value_counts().sort_index())

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
    )

    print("\nTraining binary model...")
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test).astype(int).ravel()
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC-AUC: {auc:.4f}")
    except Exception as e:
        print("ROC-AUC failed:", e)

    model_path = MODELS_DIR / "binary_model.cbm"
    model.save_model(model_path)

    print(f"\nModel saved to: {model_path}")

    print("\n=== FEATURE IMPORTANCE (top 15) ===")
    importances = model.get_feature_importance()
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    print(fi.head(15).to_string(index=False))

    fi_path = ARTIFACTS_DIR / "binary_feature_importance.csv"
    fi.to_csv(fi_path, index=False, encoding="utf-8")
    print(f"\nFeature importance saved to: {fi_path}")


if __name__ == "__main__":
    main()