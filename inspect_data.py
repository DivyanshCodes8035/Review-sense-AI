import pandas as pd
import glob
import os

# Find CSV files
files = glob.glob("dataset/*.csv")

print("CSV files found:")
for file in files:
    print("-", os.path.basename(file))

if not files:
    print("\n❌ No CSV file found!")
    print("Make sure the CSV is inside the dataset folder.")
    exit()

# Load the first CSV
file = files[0]
df = pd.read_csv(file)

print("\nUsing file:", os.path.basename(file))

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].dropna().unique()[:20])