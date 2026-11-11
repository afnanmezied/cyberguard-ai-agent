"""Train and evaluate the CyberGuard Logistic Regression baseline locally.

Run from the project root: python ml/train_logistic_regression.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from paths import BASE_DIR, TEST_DATA, TRAIN_DATA, VALIDATION_DATA

NUMERIC_FEATURES = [
    "url_length", "num_dots", "has_https", "has_ip", "num_subdirs",
    "num_params", "suspicious_words", "special_char_count", "digits_count", "entropy",
]
CATEGORICAL_FEATURES = ["tld"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OUTPUT_DIR = BASE_DIR / "models" / "logistic_regression"


def build_model():
    """Match the original Kaggle preprocessing and model settings."""
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def evaluate(model, df, split_name):
    """Report phishing-class metrics, and return a one-row summary."""
    y_true = df["label"]
    predictions = model.predict(df[FEATURES])
    print(f"\n========== {split_name.upper()} RESULTS ==========")
    print(classification_report(
        y_true, predictions, target_names=["Benign", "Phishing"],
        digits=4, zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
    return {
        "Model": "Logistic Regression",
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Phishing Precision": precision_score(y_true, predictions, zero_division=0),
        "Phishing Recall": recall_score(y_true, predictions, zero_division=0),
        "Phishing F1": f1_score(y_true, predictions, zero_division=0),
        "Missed Phishing": int(((y_true == 1) & (predictions == 0)).sum()),
    }


def main():
    train_df = pd.read_csv(TRAIN_DATA, keep_default_na=False)
    val_df = pd.read_csv(VALIDATION_DATA, keep_default_na=False)
    test_df = pd.read_csv(TEST_DATA, keep_default_na=False)
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))

    model = build_model()
    print("\nTraining Logistic Regression...")
    model.fit(train_df[FEATURES], train_df["label"])
    print("Training completed!")

    validation_results = evaluate(model, val_df, "validation")
    # Test results are for reproducing the experiment, NOT for model selection.
    test_results = evaluate(model, test_df, "test")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "logistic_regression.joblib")
    pd.DataFrame([validation_results, test_results]).to_csv(
        OUTPUT_DIR / "metrics.csv", index=False,
    )
    print("\nSaved model and metrics to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
