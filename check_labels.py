import pandas as pd
import glob

# Load dataset
file = glob.glob("dataset/*.csv")[0]
df = pd.read_csv(file)

print("===== DATASET LABEL ANALYSIS =====")

# --------------------------------------------------
# 1. Sentiment distribution
# --------------------------------------------------

print("\n--- Sentiment Distribution ---")

print(
    df["Sentiment"]
    .value_counts()
)

print("\n--- Sentiment Percentage ---")

print(
    (df["Sentiment"].value_counts(normalize=True) * 100)
    .round(2)
)

# --------------------------------------------------
# 2. Rate distribution
# --------------------------------------------------

print("\n--- Rate Values ---")

print(
    df["Rate"]
    .value_counts(dropna=False)
    .sort_index()
)

# --------------------------------------------------
# 3. Rate vs Sentiment
# --------------------------------------------------

print("\n--- Rate vs Sentiment ---")

rate_sentiment = pd.crosstab(
    df["Rate"],
    df["Sentiment"]
)

print(rate_sentiment)

# --------------------------------------------------
# 4. Percentage of sentiment inside each Rate
# --------------------------------------------------

print("\n--- Sentiment Percentage Within Each Rate ---")

rate_sentiment_percent = pd.crosstab(
    df["Rate"],
    df["Sentiment"],
    normalize="index"
) * 100

print(
    rate_sentiment_percent.round(2)
)

# --------------------------------------------------
# 5. Missing text
# --------------------------------------------------

print("\n--- Missing Text ---")

print(
    df[["Review", "Summary"]]
    .isna()
    .sum()
)

# --------------------------------------------------
# 6. Combined usable text
# --------------------------------------------------

combined_text = (
    df["Review"].fillna("").astype(str).str.strip()
    + " "
    + df["Summary"].fillna("").astype(str).str.strip()
).str.strip()

print("\n--- Combined Text ---")

print("Empty combined text:", (combined_text == "").sum())
print("Usable combined text:", (combined_text != "").sum())

# --------------------------------------------------
# 7. Conflicting Review + Sentiment count
# --------------------------------------------------

review_conflicts = (
    df.dropna(subset=["Review"])
      .groupby("Review")["Sentiment"]
      .nunique()
)

print("\n--- Review Conflicts ---")

print(
    "Reviews having multiple sentiment labels:",
    (review_conflicts > 1).sum()
)

print("\n===== ANALYSIS COMPLETE =====")