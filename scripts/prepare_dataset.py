"""Create cleaned data and an exact-text-grouped train/test split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing import EXPECTED_LABELS, prepare_dataset


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "Dataset" / "Dataset-SA.csv"
OUTPUT_DIR = ROOT / "Dataset" / "processed"
RANDOM_SEED = 42
TEST_FRACTION = 0.20


def main() -> None:
    raw = pd.read_csv(INPUT_PATH, low_memory=False)
    data, cleaning = prepare_dataset(raw)
    if set(data["label"].unique()) != set(EXPECTED_LABELS):
        raise ValueError("All three expected sentiment classes must remain after cleaning.")

    # The normalized final model input is the grouping key. This keeps identical
    # inputs (including rows with conflicting labels) on one side of the split.
    groups = data["text"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(data["text"], data["label"], groups))
    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    overlap = set(train["text"]).intersection(test["text"])
    if overlap:
        raise RuntimeError(f"Leakage check failed: {len(overlap)} text groups cross the split.")
    if set(train["label"].unique()) != set(EXPECTED_LABELS) or set(test["label"].unique()) != set(EXPECTED_LABELS):
        raise RuntimeError("The split does not contain all three sentiment classes on both sides.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "cleaned_reviews.csv", index=False, encoding="utf-8")
    train.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    test.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")

    report = {
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "random_seed": RANDOM_SEED,
        "split_method": "StratifiedGroupKFold (first of 5 folds)",
        "requested_test_fraction": TEST_FRACTION,
        "actual_test_fraction": round(len(test) / len(data), 6),
        "group_key": "normalized combined Review + Summary text",
        "text_separator": " [SEP] ",
        "labels": list(EXPECTED_LABELS),
        "cleaning": cleaning,
        "cleaned_class_counts": {k: int(v) for k, v in data["label"].value_counts().sort_index().items()},
        "train_rows": len(train),
        "test_rows": len(test),
        "train_class_counts": {k: int(v) for k, v in train["label"].value_counts().sort_index().items()},
        "test_class_counts": {k: int(v) for k, v in test["label"].value_counts().sort_index().items()},
        "unique_text_groups": int(data["text"].nunique()),
        "overlapping_text_groups": len(overlap),
        "excluded_features": ["Rate", "product_name", "product_price"],
    }
    (OUTPUT_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
