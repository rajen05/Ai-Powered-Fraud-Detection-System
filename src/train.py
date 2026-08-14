"""
train.py

Trains three models (Logistic Regression baseline, Random Forest, XGBoost)
on the fraud dataset, using class weighting as the imbalance strategy
(a second, SMOTE-based run is done separately in the notebook for comparison).
Saves the best-performing model to models/fraud_model.pkl.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocess import load_data, scale_features, split_data
from evaluate import evaluate_model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    # scale_pos_weight approximates class_weight='balanced' for XGBoost
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=fraud_ratio,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    df = load_data()
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    results = {}

    print("\n--- Logistic Regression ---")
    lr = train_logistic_regression(X_train, y_train)
    results["Logistic Regression"] = evaluate_model(lr, X_test, y_test)

    print("\n--- Random Forest ---")
    rf = train_random_forest(X_train, y_train)
    results["Random Forest"] = evaluate_model(rf, X_test, y_test)

    print("\n--- XGBoost ---")
    xgb = train_xgboost(X_train, y_train)
    results["XGBoost"] = evaluate_model(xgb, X_test, y_test)

    # Pick best model by F1 score (swap for recall if you want to weight
    # "catching fraud" even higher than balanced F1)
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}[best_name]

    print(f"\nBest model: {best_name} (F1 = {results[best_name]['f1']:.4f})")
    joblib.dump(best_model, "models/fraud_model.pkl")
    print("Saved to models/fraud_model.pkl")


if __name__ == "__main__":
    main()
