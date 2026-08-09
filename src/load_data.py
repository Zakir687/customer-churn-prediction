from pathlib import Path
import pandas as pd

# Project root folder
ROOT = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = ROOT / "data" / "Telco_Customer_Churn.csv"

print("Looking for dataset at:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nChurn distribution:")
print(df["Churn"].value_counts())