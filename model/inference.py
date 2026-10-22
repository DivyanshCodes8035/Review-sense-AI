"""Load the saved ReviewSense AI model and predict review sentiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from scripts.preprocessing import combine_review_text


MODEL_PATH = Path(__file__).resolve().with_name("reviewsense_tfidf_logreg.joblib")


class ReviewSensePredictor:
    """Reusable predictor; load the model once and reuse it for many reviews."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Run scripts/evaluate_baseline_test.py first."
            )

        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.neutral_threshold = float(artifact["neutral_threshold"])
        self.labels = list(artifact["labels"])

    def predict(self, review: Any = "", summary: Any = "") -> dict[str, Any]:
        """Predict sentiment from Review and Summary using training-time rules."""
        text = combine_review_text(review, summary)
        if not text:
            raise ValueError("Provide a non-empty product review or summary.")

        vectorizer = self.pipeline.named_steps["tfidf"]
        classifier = self.pipeline.named_steps["classifier"]
        features = vectorizer.transform([text])
        probabilities = classifier.predict_proba(features)[0]
        classes = list(classifier.classes_)
        scores = {label: float(probabilities[classes.index(label)]) for label in self.labels}

        # Match the neutral-threshold decision rule used in final evaluation.
        if scores["neutral"] >= self.neutral_threshold:
            sentiment = "neutral"
        else:
            sentiment = max(("negative", "positive"), key=scores.__getitem__)

        # For a linear classifier, each TF-IDF value multiplied by its
        # class-weight coefficient contributes to that class's decision score.
        # Compare against another eligible sentiment under the decision rule.
        competing_label = max(
            (label for label in ("negative", "positive") if label != sentiment),
            key=scores.__getitem__,
        )
        chosen_index = classes.index(sentiment)
        competing_index = classes.index(competing_label)
        coefficient_difference = classifier.coef_[chosen_index] - classifier.coef_[competing_index]
        feature_names = vectorizer.get_feature_names_out()
        feature_row = features.getrow(0)
        contributions = feature_row.data * coefficient_difference[feature_row.indices]

        supporting_terms = []
        opposing_terms = []
        for feature_index, contribution in zip(feature_row.indices, contributions):
            term = str(feature_names[feature_index])
            # The literal separator joins Review and Summary; it is not review evidence.
            if "sep" in term.split() or abs(float(contribution)) < 1e-6:
                continue
            entry = {"term": term, "contribution": round(float(contribution), 4)}
            if contribution > 0:
                supporting_terms.append(entry)
            else:
                opposing_terms.append(entry)

        supporting_terms.sort(key=lambda item: item["contribution"], reverse=True)
        opposing_terms.sort(key=lambda item: item["contribution"])

        return {
            "sentiment": sentiment,
            # This is the model's estimated probability for the predicted class.
            # It has not been calibrated, so avoid presenting it as certainty.
            "confidence_estimate": round(scores[sentiment], 4),
            "class_probability_estimates": {label: round(score, 4) for label, score in scores.items()},
            "explanation": {
                "compared_with": competing_label,
                "supporting_terms": supporting_terms[:5],
                "opposing_terms": opposing_terms[:5],
            },
        }


def main() -> None:
    predictor = ReviewSensePredictor()
    result = predictor.predict(
        review="The product works well and feels like good value.",
        summary="Good quality",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
