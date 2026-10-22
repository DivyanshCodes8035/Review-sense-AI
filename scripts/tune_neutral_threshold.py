"""Tune the neutral decision threshold using the grouped validation split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "Dataset" / "processed" / "train.csv"
REPORT_PATH = ROOT / "model" / "reports" / "neutral_threshold_tuning.json"
LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


def collect_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Return the same class metrics for every threshold tested."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main() -> None:
    data = pd.read_csv(TRAIN_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError(f"{TRAIN_PATH} must contain 'text' and 'label' columns.")

    # Recreate the same validation fold used in the previous experiments.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_indices, validation_indices = next(
        splitter.split(data["text"], data["label"], groups=data["text"])
    )
    train = data.iloc[train_indices]
    validation = data.iloc[validation_indices]
    if set(train["text"]).intersection(validation["text"]):
        raise RuntimeError("Validation leakage check failed: identical text is in both splits.")

    # Keep the stronger unweighted setting from the controlled comparison.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train["text"])
    validation_features = vectorizer.transform(validation["text"])
    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    classifier.fit(train_features, train["label"])

    # Start with ordinary highest-probability prediction as the reference.
    probabilities = classifier.predict_proba(validation_features)
    class_positions = {label: list(classifier.classes_).index(label) for label in LABELS}
    neutral_probability = probabilities[:, class_positions["neutral"]]
    negative_probability = probabilities[:, class_positions["negative"]]
    positive_probability = probabilities[:, class_positions["positive"]]
    default_predictions = classifier.predict(validation_features)

    # If neutral probability clears the threshold, predict neutral. Otherwise,
    # choose whichever of positive or negative has the larger probability.
    # The best threshold is selected by validation macro F1, which treats all
    # three sentiment classes equally.
    candidates = np.linspace(0.05, 0.95, 91)
    best_threshold = None
    best_macro_f1 = -1.0
    best_neutral_f1 = -1.0
    best_predictions = None
    for threshold in candidates:
        predictions = np.where(
            neutral_probability >= threshold,
            "neutral",
            np.where(negative_probability >= positive_probability, "negative", "positive"),
        )
        macro_f1 = f1_score(validation["label"], predictions, labels=LABELS, average="macro", zero_division=0)
        neutral_f1 = f1_score(
            validation["label"], predictions, labels=["neutral"], average="macro", zero_division=0
        )
        # If macro F1 ties, prefer the threshold with better neutral F1.
        if (macro_f1, neutral_f1) > (best_macro_f1, best_neutral_f1):
            best_threshold = float(threshold)
            best_macro_f1 = float(macro_f1)
            best_neutral_f1 = float(neutral_f1)
            best_predictions = predictions

    best_metrics = collect_metrics(validation["label"], best_predictions)
    default_metrics = collect_metrics(validation["label"], default_predictions)
    report = {
        "experiment": "Neutral probability threshold tuning",
        "random_seed": RANDOM_SEED,
        "validation_split": "first fold of 5-fold StratifiedGroupKFold on train.csv",
        "group_key": "exact combined text",
        "overlapping_text_groups": 0,
        "selection_metric": "validation macro F1; neutral F1 breaks ties",
        "threshold_search": {"start": 0.05, "end": 0.95, "step": 0.01},
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "default_argmax": default_metrics,
        "best_neutral_threshold": best_threshold,
        "best_threshold_metrics": best_metrics,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Training rows: {len(train):,}; validation rows: {len(validation):,}")
    print("Validation text groups shared with training: 0")
    print("\nDefault unweighted Logistic Regression:")
    print(f"  Accuracy: {default_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {default_metrics['macro_f1']:.4f}")
    neutral = default_metrics["classification_report"]["neutral"]
    print(f"  Neutral precision / recall / F1: {neutral['precision']:.4f} / {neutral['recall']:.4f} / {neutral['f1-score']:.4f}")
    print(f"\nBest neutral threshold: {best_threshold:.2f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {best_metrics['macro_f1']:.4f}")
    neutral = best_metrics["classification_report"]["neutral"]
    print(f"  Neutral precision / recall / F1: {neutral['precision']:.4f} / {neutral['recall']:.4f} / {neutral['f1-score']:.4f}")
    print("\nPer-class results at the selected threshold:")
    print(
        classification_report(
            validation["label"],
            best_predictions,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
        )
    )
    print(f"Detailed report saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
