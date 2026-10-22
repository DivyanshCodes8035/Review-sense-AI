"""Train and evaluate the first ReviewSense AI model on a validation split."""

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
REPORT_PATH = ROOT / "model" / "reports" / "baseline_validation.json"

# Keep a fixed class order so the printed and saved results are easy to compare.
LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


def main() -> None:
    # We split only the training portion created earlier. test.csv stays untouched
    # until the model and settings are finalized.
    data = pd.read_csv(TRAIN_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"{TRAIN_PATH} must contain columns: {sorted(required_columns)}")

    # Group by the exact model input. Repeated text, including conflicting labels,
    # is assigned wholly to training or wholly to validation.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_indices, validation_indices = next(
        splitter.split(data["text"], data["label"], groups=data["text"])
    )
    train = data.iloc[train_indices]
    validation = data.iloc[validation_indices]

    overlap = set(train["text"]).intersection(validation["text"])
    if overlap:
        raise RuntimeError(f"Validation leakage check failed: {len(overlap)} text groups overlap.")

    # TF-IDF represents words and adjacent word pairs by how informative they are.
    # Keeping punctuation and negation in the text helps preserve sentiment cues.
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

    # Balanced class weights counter the dataset's large positive-class majority.
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    classifier.fit(train_features, train["label"])
    predictions = classifier.predict(validation_features)

    class_details = classification_report(
        validation["label"],
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report = {
        "model": "TF-IDF (word unigrams and bigrams) + Logistic Regression",
        "random_seed": RANDOM_SEED,
        "validation_split": "first fold of 5-fold StratifiedGroupKFold on train.csv",
        "group_key": "exact combined text",
        "class_weight": "balanced",
        "labels_in_order": LABELS,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_unique_texts": int(train["text"].nunique()),
        "validation_unique_texts": int(validation["text"].nunique()),
        "overlapping_text_groups": len(overlap),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "accuracy": float(accuracy_score(validation["label"], predictions)),
        "macro_f1": float(f1_score(validation["label"], predictions, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(validation["label"], predictions, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": class_details,
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            validation["label"], predictions, labels=LABELS
        ).tolist(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Validation text groups shared with training: {len(overlap)}")
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"Validation accuracy: {report['accuracy']:.4f}")
    print(f"Validation macro F1: {report['macro_f1']:.4f}")
    print(f"Validation weighted F1: {report['weighted_f1']:.4f}")
    print("\nPer-class results:")
    print(classification_report(validation["label"], predictions, labels=LABELS, target_names=LABELS, zero_division=0))
    print(f"Detailed report saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
