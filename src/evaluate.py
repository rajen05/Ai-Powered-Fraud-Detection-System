"""
evaluate.py

Evaluation helpers. Deliberately does NOT lead with accuracy: on a dataset
that's 99.83% "not fraud", a model that predicts "not fraud" every single
time scores 99.83% accuracy while catching zero fraud. Precision, recall,
F1, and PR-AUC are what actually tell you if the model works.
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Runs predictions and prints/returns the metrics that matter for
    imbalanced fraud detection.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),  # more informative than ROC-AUC here
    }

    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}  <- most meaningful metric on this imbalanced data")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return metrics
