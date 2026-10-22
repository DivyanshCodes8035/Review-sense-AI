import pandas as pd
import glob

file = glob.glob("dataset/*.csv")[0]
df = pd.read_csv(file)

# Remove rows where Review is missing
reviews = df.dropna(subset=["Review"])

# Find reviews that have more than one sentiment
conflicts = (
    reviews.groupby("Review")["Sentiment"]
    .nunique()
    .sort_values(ascending=False)
)

conflicting_reviews = conflicts[conflicts > 1]

print("===== CONFLICT CHECK =====")

print("Unique reviews:", reviews["Review"].nunique())

print("Reviews with multiple sentiments:",
      len(conflicting_reviews))

print("\nExamples:")

for review in conflicting_reviews.head(20).index:
    print("\nReview:", review)
    print(
        reviews[reviews["Review"] == review]["Sentiment"]
        .value_counts()
    )
    