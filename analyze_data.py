import pandas as pd
import glob
import os

# Find the dataset
file = glob.glob("dataset/*.csv")[0]

df = pd.read_csv(file)

print("===== DATASET INFO =====")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\n===== SENTIMENT DISTRIBUTION =====")
print(df["Sentiment"].value_counts())

print("\n===== SENTIMENT PERCENTAGE =====")
print(df["Sentiment"].value_counts(normalize=True) * 100)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE REVIEWS =====")
print("Duplicate Review rows:", df["Review"].duplicated().sum())

print("\n===== DUPLICATE SUMMARIES =====")
print("Duplicate Summary rows:", df["Summary"].duplicated().sum())

print("\n===== REVIEW LENGTH =====")
print(df["Review"].dropna().str.len().describe())

print("\n===== SUMMARY LENGTH =====")
print(df["Summary"].dropna().str.len().describe())