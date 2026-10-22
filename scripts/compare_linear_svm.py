"""Compare a LinearSVC model with the selected Logistic Regression baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "Dataset" / "processed" / "train.csv"
LOGISTIC_REPORT_PATH = ROOT / "model" / "reports" / "neutral_threshold_tuning.json"
REPORT_PATH = ROOT / "model" / "reports" / "linear_svm_comparison.json"

LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


def summarize(y_true: pd.Series, y_pred) -> dict:
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
    data = pd.read_csv(TRAIN_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError(f"{TRAIN_PATH} must contain 'text' and 'label' columns.")
    if not LOGISTIC_REPORT_PATH.exists():
        raise FileNotFoundError(f"Run the validation threshold step first: {LOGISTIC_REPORT_PATH}")

    # Reuse the exact grouped validation fold from the Logistic Regression runs.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_indices, validation_indices = next(
        splitter.split(data["text"], data["label"], groups=data["text"])
    )
    train = data.iloc[train_indices]
    validation = data.iloc[validation_indices]
    if set(train["text"]).intersection(validation["text"]):
        raise RuntimeError("Validation leakage check failed: identical text is in both splits.")

    # Keep the text representation identical so this comparison focuses on the
    # classifier. The vocabulary is learned from the training fold only.
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

    # LinearSVC is a common, efficient classifier for sparse text features.
    # It predicts labels directly; this version does not produce probabilities.
    classifier = LinearSVC(C=1.0, class_weight=None, random_state=RANDOM_SEED, max_iter=5000)
    classifier.fit(train_features, train["label"])
    predictions = classifier.predict(validation_features)

    with LOGISTIC_REPORT_PATH.open(encoding="utf-8") as report_file:
        logistic_report = json.load(report_file)
    logistic_validation = logistic_report["best_threshold_metrics"]
    svm_validation = summarize(validation["label"], predictions)

    report = {
        "experiment": "TF-IDF Logistic Regression vs. LinearSVC",
        "random_seed": RANDOM_SEED,
        "validation_split": "first fold of 5-fold StratifiedGroupKFold on train.csv",
        "group_key": "exact combined text",
        "overlapping_text_groups": 0,
        "same_tfidf_settings": True,
        "logistic_regression": {
            "decision_rule": "neutral probability threshold 0.26, chosen on this validation fold",
            "metrics": logistic_validation,
        },
        "linear_svc": {
            "C": 1.0,
            "class_weight": None,
            "probability_scores_available": False,
            "metrics": svm_validation,
        },
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Training rows: {len(train):,}; validation rows: {len(validation):,}")
    print("Validation text groups shared with training: 0")
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    print("\nSelected Logistic Regression (threshold 0.26):")
    print(f"  Macro F1: {logistic_validation['macro_f1']:.4f}")
    print(f"  Neutral F1: {logistic_validation['classification_report']['neutral']['f1-score']:.4f}")
    print("\nLinearSVC (direct class prediction):")
    print(f"  Accuracy: {svm_validation['accuracy']:.4f}")
    print(f"  Macro F1: {svm_validation['macro_f1']:.4f}")
    print(f"  Weighted F1: {svm_validation['weighted_f1']:.4f}")
    print("  Per-class results:")
    print(classification_report(validation["label"], predictions, labels=LABELS, target_names=LABELS, zero_division=0))
    print(f"Detailed comparison saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
