from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing import (
    load_data,
    split_features_target,
    make_train_test_split,
    scale_time_amount,
)
from evaluate import make_model_comparison_row, threshold_search


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "creditcard.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "random_forest_fraud_model.joblib"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "threshold_config.json"

MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_ANALYSIS_PATH = REPORTS_DIR / "threshold_analysis.csv"

RANDOM_STATE = 42


def train_logistic_regression(X_train_scaled, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_hist_gradient_boosting(X_train, y_train):
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Target distribution:")
    print(df["Class"].value_counts(normalize=True))

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_time_amount(X_train, X_test)

    print("\nTraining Logistic Regression...")
    log_reg = train_logistic_regression(X_train_scaled, y_train)
    y_proba_logreg = log_reg.predict_proba(X_test_scaled)[:, 1]

    print("Training Random Forest...")
    random_forest = train_random_forest(X_train, y_train)
    y_proba_rf = random_forest.predict_proba(X_test)[:, 1]

    print("Training HistGradientBoosting...")
    hgb = train_hist_gradient_boosting(X_train, y_train)
    y_proba_hgb = hgb.predict_proba(X_test)[:, 1]

    print("\nCreating model comparison table...")
    model_comparison = pd.DataFrame(
        [
            make_model_comparison_row(
                "Logistic Regression",
                y_test,
                y_proba_logreg,
                threshold=0.5,
            ),
            make_model_comparison_row(
                "Random Forest",
                y_test,
                y_proba_rf,
                threshold=0.5,
            ),
            make_model_comparison_row(
                "HistGradientBoosting",
                y_test,
                y_proba_hgb,
                threshold=0.5,
            ),
        ]
    )

    model_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    print("\nModel comparison:")
    print(model_comparison)

    print("\nRunning threshold search for Random Forest...")
    threshold_df = threshold_search(
        y_test,
        y_proba_rf,
        false_positive_cost=5,
        false_negative_cost=100,
    )

    threshold_df.to_csv(THRESHOLD_ANALYSIS_PATH, index=False)

    best_by_f1 = threshold_df.sort_values("f1", ascending=False).iloc[0]
    best_by_cost = threshold_df.sort_values("business_cost").iloc[0]

    selected_threshold = float(best_by_cost["threshold"])

    print("\nBest threshold by F1:")
    print(best_by_f1)

    print("\nBest threshold by business cost:")
    print(best_by_cost)

    print(f"\nSelected threshold: {selected_threshold}")

    print("\nSaving final model and config...")
    joblib.dump(random_forest, MODEL_PATH)

    if scaler is not None:
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    threshold_config = {
        "selected_model": "Random Forest",
        "selected_threshold": selected_threshold,
        "approve_threshold": 0.20,
        "review_threshold": selected_threshold,
        "decision_logic": {
            "approve": "fraud_probability < 0.20",
            "review": f"0.20 <= fraud_probability < {selected_threshold}",
            "block": f"fraud_probability >= {selected_threshold}",
        },
        "business_cost": {
            "false_positive_cost": 5,
            "false_negative_cost": 100,
        },
    }

    with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(threshold_config, file, indent=4, ensure_ascii=False)

    print("\nDone.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Threshold config saved to: {THRESHOLD_CONFIG_PATH}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_PATH}")
    print(f"Threshold analysis saved to: {THRESHOLD_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()