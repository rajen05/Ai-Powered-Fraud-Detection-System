"""
preprocess.py

Loads the raw credit card transaction data, scales the columns that need it,
and produces a stratified train/test split. Also handles class imbalance
on the TRAINING set only (never touch the test set with SMOTE — that would
leak synthetic fraud patterns into your evaluation and inflate your scores).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the raw Kaggle CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df['Class'].sum()} of which are fraud "
          f"({df['Class'].mean() * 100:.3f}%).")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale 'Amount' and 'Time'. The V1-V28 columns come from PCA and are
    already roughly standardized, so we leave them alone.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified split so the test set keeps the same ~0.17% fraud ratio
    as the full dataset. A plain random split can accidentally leave you
    with a test set that has almost no fraud examples in it.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def balance_with_smote(X_train, y_train, random_state: int = 42):
    """
    Oversample the minority (fraud) class in the TRAINING set only.
    Returns a new, balanced X_train/y_train. X_test/y_test must stay untouched.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled


if __name__ == "__main__":
    df = load_data()
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
