"""Compare Logistic Regression with and without class weighting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "Dataset" / "processed" / "train.csv"
REPORT_PATH = ROOT / "model" / "reports" / "class_weight_comparison.json"
LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


def evaluate_variant(
    name: str,
    class_weight: str | None,
    train_features,
    train_labels: pd.Series,
    validation_features,
    validation_labels: pd.Series,
) -> dict:
    """Fit one classifier and collect comparable validation metrics."""
    classifier = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(validation_features)

    per_class = classification_report(
        validation_labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "name": name,
        "class_weight": class_weight,
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "macro_f1": float(f1_score(validation_labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(validation_labels, predictions, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": per_class,
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            validation_labels, predictions, labels=LABELS
        ).tolist(),
    }


def main() -> None:
    data = pd.read_csv(TRAIN_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError(f"{TRAIN_PATH} must contain 'text' and 'label' columns.")

    # Use precisely the same grouped split and TF-IDF settings for both models.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_indices, validation_indices = next(
        splitter.split(data["text"], data["label"], groups=data["text"])
    )
    train = data.iloc[train_indices]
    validation = data.iloc[validation_indices]
    if set(train["text"]).intersection(validation["text"]):
        raise RuntimeError("Validation leakage check failed: identical text is in both splits.")

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

    variants = [
        evaluate_variant(
            "balanced", "balanced", train_features, train["label"], validation_features, validation["label"]
        ),
        evaluate_variant(
            "no_class_weights", None, train_features, train["label"], validation_features, validation["label"]
        ),
    ]

    report = {
        "experiment": "Effect of Logistic Regression class weights",
        "random_seed": RANDOM_SEED,
        "validation_split": "first fold of 5-fold StratifiedGroupKFold on train.csv",
        "shared_text_groups": 0,
        "same_tfidf_features_for_both_variants": True,
        "labels_in_order": LABELS,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "results": variants,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Training rows: {len(train):,}; validation rows: {len(validation):,}")
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    print("Text groups shared across splits: 0")
    for result in variants:
        neutral = result["classification_report"]["neutral"]
        print(f"\n{result['name']} class weights:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Macro F1: {result['macro_f1']:.4f}")
        print(f"  Weighted F1: {result['weighted_f1']:.4f}")
        print(
            "  Neutral precision: "
            f"{neutral['precision']:.4f}; recall: {neutral['recall']:.4f}; F1: {neutral['f1-score']:.4f}"
        )
        print("  Per-class precision / recall / F1:")
        for label in LABELS:
            metrics = result["classification_report"][label]
            print(
                f"    {label}: {metrics['precision']:.4f} / "
                f"{metrics['recall']:.4f} / {metrics['f1-score']:.4f}"
            )
    print(f"\nDetailed comparison saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
