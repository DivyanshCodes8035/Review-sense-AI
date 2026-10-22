"""Train the selected baseline and evaluate it once on the untouched test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "Dataset" / "processed" / "train.csv"
TEST_PATH = ROOT / "Dataset" / "processed" / "test.csv"
MODEL_PATH = ROOT / "model" / "reviewsense_tfidf_logreg.joblib"
REPORT_PATH = ROOT / "model" / "reports" / "baseline_test_evaluation.json"

LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42
NEUTRAL_THRESHOLD = 0.26  # Chosen using validation data, never the test data.


def apply_neutral_threshold(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Predict neutral above its selected threshold; otherwise choose pos/neg."""
    class_positions = {label: list(classes).index(label) for label in LABELS}
    neutral_probability = probabilities[:, class_positions["neutral"]]
    negative_probability = probabilities[:, class_positions["negative"]]
    positive_probability = probabilities[:, class_positions["positive"]]
    return np.where(
        neutral_probability >= NEUTRAL_THRESHOLD,
        "neutral",
        np.where(negative_probability >= positive_probability, "negative", "positive"),
    )


def summarize(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if not {"text", "label"}.issubset(train.columns) or not {"text", "label"}.issubset(test.columns):
        raise ValueError("Both train.csv and test.csv must contain 'text' and 'label' columns.")

    # Protect the final estimate: the exact combined text used for modeling must
    # not occur in both datasets.
    overlapping_text = set(train["text"]).intersection(test["text"])
    if overlapping_text:
        raise RuntimeError(f"Test leakage check failed: {len(overlapping_text)} text groups overlap.")

    # Refit on all training rows after selecting settings with validation.
    # TF-IDF learns its vocabulary from training text only.
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=300_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            ),
        ]
    )
    model.fit(train["text"], train["label"])

    probabilities = model.predict_proba(test["text"])
    predictions = apply_neutral_threshold(probabilities, model.named_steps["classifier"].classes_)
    metrics = summarize(test["label"], predictions)
    report = {
        "model": "TF-IDF word unigrams/bigrams + unweighted Logistic Regression",
        "threshold_policy": "neutral if P(neutral) >= 0.26; otherwise choose larger P(negative), P(positive)",
        "neutral_threshold_selected_on": "validation split only",
        "random_seed": RANDOM_SEED,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_unique_texts": int(train["text"].nunique()),
        "test_unique_texts": int(test["text"].nunique()),
        "overlapping_text_groups": len(overlapping_text),
        "vocabulary_size": int(len(model.named_steps["tfidf"].vocabulary_)),
        "test_metrics": metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Store the fitted text model and its decision rule together for later reuse.
    joblib.dump({"pipeline": model, "neutral_threshold": NEUTRAL_THRESHOLD, "labels": LABELS}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    neutral = metrics["classification_report"]["neutral"]
    print(f"Training rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Test text groups shared with training: {len(overlapping_text)}")
    print(f"TF-IDF vocabulary size: {len(model.named_steps['tfidf'].vocabulary_):,}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {metrics['weighted_f1']:.4f}")
    print(
        "Test neutral precision / recall / F1: "
        f"{neutral['precision']:.4f} / {neutral['recall']:.4f} / {neutral['f1-score']:.4f}"
    )
    print("\nPer-class results:")
    print(classification_report(test["label"], predictions, labels=LABELS, target_names=LABELS, zero_division=0))
    print(f"\nModel candidate saved to: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Test report saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
