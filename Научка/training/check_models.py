#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "training" / "artifacts"
MODELS_DIR = ROOT / "training" / "models"

VAL_PATH = ARTIFACTS_DIR / "val.csv"
TEST_PATH = ARTIFACTS_DIR / "test.csv"
FULL_PATH = ARTIFACTS_DIR / "dataset_full.csv"

BINARY_MODEL_PATH = MODELS_DIR / "binary_model.cbm"
MULTICLASS_MODEL_PATH = MODELS_DIR / "multiclass_model.cbm"


KNOWN_CATEGORICAL = {
    "payer_client_id",
    "payer_bic",
    "beneficiary_bic",
    "trn_type",
    "narrative",
}


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_X_for_model(df: pd.DataFrame, model: CatBoostClassifier) -> pd.DataFrame:
    """
    Строит X строго в том порядке и с теми именами колонок,
    которые модель видела при обучении.
    """
    feature_names = model.feature_names_
    X = df[feature_names].copy()

    for col in feature_names:
        if col in KNOWN_CATEGORICAL:
            X[col] = X[col].fillna("NA").astype(str)

    return X


def main():
    print("Loading saved models...")
    binary_model = CatBoostClassifier()
    binary_model.load_model(BINARY_MODEL_PATH)

    multiclass_model = CatBoostClassifier()
    multiclass_model.load_model(MULTICLASS_MODEL_PATH)

    print("Binary model feature order:")
    print(binary_model.feature_names_)

    print("\nMulticlass model feature order:")
    print(multiclass_model.feature_names_)

    print("\nLoading validation/test data...")
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)
    full_df = load_csv(FULL_PATH)

    # ---------------- BINARY ----------------
    print("\n=== BINARY MODEL ON TEST ===")
    X_test_bin = build_X_for_model(test_df, binary_model)
    y_test_bin = test_df["target_binary"].astype(int)

    y_pred_bin = binary_model.predict(X_test_bin).astype(int).ravel()
    y_proba_bin = binary_model.predict_proba(X_test_bin)[:, 1]

    print(classification_report(y_test_bin, y_pred_bin, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_bin, y_pred_bin))
    print(f"ROC-AUC: {roc_auc_score(y_test_bin, y_proba_bin):.4f}")

    test_df_bin_out = test_df.copy()
    test_df_bin_out["pred_binary"] = y_pred_bin
    test_df_bin_out["proba_anomaly"] = y_proba_bin

    cols_show = [
        "timestamp",
        "amount",
        "payer_client_id",
        "beneficiary_bic",
        "target_binary",
        "base_label",
        "proba_anomaly",
        "pred_binary",
    ]
    print("\nTop-10 most suspicious transactions by binary model:")
    print(
        test_df_bin_out.sort_values("proba_anomaly", ascending=False)[cols_show]
        .head(10)
        .to_string(index=False)
    )

    # ---------------- MULTICLASS ----------------
    print("\n=== MULTICLASS MODEL ON STRATIFIED ANOMALOUS SPLIT ===")

    mc_df = full_df[full_df["target_multiclass"] != 0].copy()
    mc_df["target_multiclass"] = mc_df["target_multiclass"].astype(int)

    # Делаем тот же тип split, что и в train_multiclass.py
    _, temp_df = train_test_split(
        mc_df,
        test_size=0.30,
        random_state=42,
        stratify=mc_df["target_multiclass"],
    )
    _, test_mc_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["target_multiclass"],
    )

    X_test_mc = build_X_for_model(test_mc_df, multiclass_model)
    y_test_mc = test_mc_df["target_multiclass"].astype(int)

    y_pred_mc = multiclass_model.predict(X_test_mc).astype(int).ravel()
    proba_mc = multiclass_model.predict_proba(X_test_mc)

    print(classification_report(y_test_mc, y_pred_mc, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_mc, y_pred_mc))

    test_mc_df_out = test_mc_df.copy()
    test_mc_df_out["pred_multiclass"] = y_pred_mc

    # Классы у вас 1..4, а индексы массива вероятностей 0..3
    test_mc_df_out["proba_class_1"] = proba_mc[:, 0]
    test_mc_df_out["proba_class_2"] = proba_mc[:, 1]
    test_mc_df_out["proba_class_3"] = proba_mc[:, 2]
    test_mc_df_out["proba_class_4"] = proba_mc[:, 3]

    cols_mc = [
        "timestamp",
        "amount",
        "payer_client_id",
        "beneficiary_bic",
        "base_label",
        "pred_multiclass",
        "proba_class_1",
        "proba_class_2",
        "proba_class_3",
        "proba_class_4",
    ]
    print("\nExamples of multiclass predictions:")
    print(test_mc_df_out[cols_mc].head(10).to_string(index=False))

    out_bin = ARTIFACTS_DIR / "test_binary_predictions.csv"
    out_mc = ARTIFACTS_DIR / "test_multiclass_predictions.csv"

    test_df_bin_out.to_csv(out_bin, index=False, encoding="utf-8")
    test_mc_df_out.to_csv(out_mc, index=False, encoding="utf-8")

    print(f"\nSaved binary predictions to: {out_bin}")
    print(f"Saved multiclass predictions to: {out_mc}")


if __name__ == "__main__":
    main()