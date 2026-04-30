from pathlib import Path
import json

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "random_forest_fraud_model.joblib"
THRESHOLD_CONFIG_PATH = ROOT_DIR / "models" / "threshold_config.json"
DATA_PATH = ROOT_DIR / "data" / "raw" / "creditcard.csv"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run `python src/train.py` first."
        )

    return joblib.load(MODEL_PATH)


def load_threshold_config():
    if not THRESHOLD_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Threshold config not found: {THRESHOLD_CONFIG_PATH}. "
            "Run `python src/train.py` first."
        )

    with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def make_decision(fraud_probability: float, config: dict) -> str:
    approve_threshold = config["approve_threshold"]
    selected_threshold = config["selected_threshold"]

    if fraud_probability < approve_threshold:
        return "approve"

    if fraud_probability < selected_threshold:
        return "review"

    return "block"


def predict_transaction(transaction: dict) -> dict:
    model = load_model()
    config = load_threshold_config()

    X = pd.DataFrame([transaction])

    fraud_probability = float(model.predict_proba(X)[:, 1][0])
    decision = make_decision(fraud_probability, config)

    return {
        "fraud_probability": round(fraud_probability, 6),
        "decision": decision,
    }


def load_example_transaction() -> dict:
    """
    Load first transaction from raw dataset as an example.

    The Class column is removed because it is the target.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Put creditcard.csv into data/raw/creditcard.csv"
        )

    df = pd.read_csv(DATA_PATH)

    example = df.drop(columns=["Class"]).iloc[0].to_dict()
    return example


if __name__ == "__main__":
    example_transaction = load_example_transaction()
    prediction = predict_transaction(example_transaction)

    print("Example prediction:")
    print(prediction)