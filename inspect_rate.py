import pandas as pd
import glob

# Load dataset
file = glob.glob("dataset/*.csv")[0]
df = pd.read_csv(file)

print("===== RATE COLUMN INSPECTION =====")

# --------------------------------------------------
# 1. Show every unique Rate value
# --------------------------------------------------

print("\n--- All Rate Values ---")

print(
    df["Rate"]
    .value_counts(dropna=False)
)

# --------------------------------------------------
# 2. Expected numeric ratings
# --------------------------------------------------

expected_rates = {"1", "2", "3", "4", "5"}

rate_as_string = (
    df["Rate"]
    .astype(str)
    .str.strip()
)

invalid_mask = ~rate_as_string.isin(expected_rates)

invalid_rows = df[invalid_mask]

print("\n--- Invalid Rate Values ---")

print("Number of invalid Rate rows:", len(invalid_rows))

print(
    "\nInvalid Rate value counts:"
)

print(
    invalid_rows["Rate"]
    .value_counts()
)

# --------------------------------------------------
# 3. Inspect actual malformed rows
# --------------------------------------------------

print("\n--- Sample Invalid Rows ---")

columns_to_show = [
    "product_name",
    "product_price",
    "Rate",
    "Review",
    "Summary",
    "Sentiment"
]

print(
    invalid_rows[columns_to_show]
    .head(20)
    .to_string(index=False)
)

# --------------------------------------------------
# 4. Check whether invalid Rate rows look shifted
# --------------------------------------------------

print("\n--- Invalid Rate + Sentiment ---")

print(
    pd.crosstab(
        invalid_rows["Rate"],
        invalid_rows["Sentiment"]
    )
)

# --------------------------------------------------
# 5. Check valid rating vs sentiment
# --------------------------------------------------

valid_rows = df[~invalid_mask].copy()

print("\n--- Valid Rate vs Sentiment ---")

print(
    pd.crosstab(
        valid_rows["Rate"],
        valid_rows["Sentiment"]
    )
)

print("\n--- Percentage ---")

print(
    (
        pd.crosstab(
            valid_rows["Rate"],
            valid_rows["Sentiment"],
            normalize="index"
        ) * 100
    ).round(2)
)

print("\n===== INSPECTION COMPLETE =====")