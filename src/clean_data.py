from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco_Customer_Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna()

print("\nShape after cleaning:")
print(df.shape)

# Remove customerID because it is not useful for prediction
df = df.drop(columns=["customerID"])

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nCleaned Dataset Shape:")
print(df.shape)

# Save cleaned dataset
OUTPUT_PATH = ROOT / "data" / "cleaned_churn_data.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print(OUTPUT_PATH)