"""Reusable text preparation for ReviewSense AI."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


TEXT_COLUMNS = ("Review", "Summary")
LABEL_COLUMN = "Sentiment"
EXPECTED_LABELS = ("negative", "neutral", "positive")
TEXT_SEPARATOR = " [SEP] "


def clean_text_value(value: Any) -> str:
    """Normalize Unicode and whitespace without removing sentiment cues."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def combine_review_text(review: Any, summary: Any) -> str:
    """Combine review and summary consistently for training and inference."""
    review_text = clean_text_value(review)
    summary_text = clean_text_value(summary)
    parts = [part for part in (review_text, summary_text) if part]
    return TEXT_SEPARATOR.join(parts)


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate, clean, and label input rows while preserving useful examples."""
    missing_columns = set((*TEXT_COLUMNS, LABEL_COLUMN, "Rate")) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    frame = raw.copy()
    frame["text"] = [combine_review_text(a, b) for a, b in zip(frame["Review"], frame["Summary"])]
    frame["label"] = frame[LABEL_COLUMN].astype("string").str.strip().str.casefold()

    # The known shifted rows have product names in Rate; do not treat these as reviews.
    numeric_rate = pd.to_numeric(frame["Rate"], errors="coerce")
    malformed_mask = numeric_rate.isna() | ~numeric_rate.isin([1, 2, 3, 4, 5])
    empty_text_mask = frame["text"].eq("")
    unknown_label_mask = ~frame["label"].isin(EXPECTED_LABELS)
    drop_mask = malformed_mask | empty_text_mask | unknown_label_mask

    counts = {
        "input_rows": int(len(frame)),
        "malformed_rate_rows_removed": int(malformed_mask.sum()),
        "empty_text_rows_removed": int((empty_text_mask & ~malformed_mask).sum()),
        "unknown_label_rows_removed": int((unknown_label_mask & ~malformed_mask & ~empty_text_mask).sum()),
        "rows_removed_total": int(drop_mask.sum()),
    }
    frame = frame.loc[~drop_mask, ["text", "label"]].reset_index(drop=True)
    counts["usable_rows"] = int(len(frame))
    return frame, counts

