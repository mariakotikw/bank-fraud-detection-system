from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def load_data(path: str | Path) -> pd.DataFrame:
    """Load raw credit card fraud dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Put creditcard.csv into data/raw/creditcard.csv"
        )

    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    """Split dataframe into features and target."""
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' was not found.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


def make_train_test_split(X, y, test_size: float = 0.2):
    """Make stratified train/test split for imbalanced classification."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def scale_time_amount(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scale Time and Amount columns.

    Scaling is used for Logistic Regression.
    Tree-based models are trained on unscaled features.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    cols_to_scale = [col for col in ["Time", "Amount"] if col in X_train.columns]

    if not cols_to_scale:
        return X_train_scaled, X_test_scaled, None

    scaler = StandardScaler()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(
        X_train_scaled[cols_to_scale]
    )
    X_test_scaled[cols_to_scale] = scaler.transform(
        X_test_scaled[cols_to_scale]
    )

    return X_train_scaled, X_test_scaled, scaler