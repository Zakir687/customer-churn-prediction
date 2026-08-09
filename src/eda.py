from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_churn_data.csv"
REPORTS_PATH = ROOT / "reports"

REPORTS_PATH.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Create ONE figure containing FOUR graphs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ------------------------------------------------
# Graph 1: Customer Churn Distribution
# ------------------------------------------------

churn_counts = df["Churn"].value_counts()

axes[0, 0].bar(
    churn_counts.index,
    churn_counts.values
)

axes[0, 0].set_title("Customer Churn Distribution")
axes[0, 0].set_xlabel("Churn")
axes[0, 0].set_ylabel("Number of Customers")


# ------------------------------------------------
# Graph 2: Churn by Contract Type
# ------------------------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

contract_churn.plot(
    kind="bar",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].set_xlabel("Contract Type")
axes[0, 1].set_ylabel("Number of Customers")
axes[0, 1].tick_params(axis="x", rotation=15)


# ------------------------------------------------
# Graph 3: Monthly Charges by Churn
# ------------------------------------------------

for status in ["No", "Yes"]:

    data = df[df["Churn"] == status]

    axes[1, 0].hist(
        data["MonthlyCharges"],
        bins=25,
        alpha=0.6,
        label=status
    )

axes[1, 0].set_title("Monthly Charges by Churn")
axes[1, 0].set_xlabel("Monthly Charges")
axes[1, 0].set_ylabel("Number of Customers")
axes[1, 0].legend(title="Churn")


# ------------------------------------------------
# Graph 4: Tenure by Churn
# ------------------------------------------------

for status in ["No", "Yes"]:

    data = df[df["Churn"] == status]

    axes[1, 1].hist(
        data["tenure"],
        bins=25,
        alpha=0.6,
        label=status
    )

axes[1, 1].set_title("Customer Tenure by Churn")
axes[1, 1].set_xlabel("Tenure (Months)")
axes[1, 1].set_ylabel("Number of Customers")
axes[1, 1].legend(title="Churn")


# ------------------------------------------------
# Main dashboard title
# ------------------------------------------------

fig.suptitle(
    "Customer Churn Exploratory Data Analysis",
    fontsize=18
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save as ONE image
OUTPUT_PATH = REPORTS_PATH / "customer_churn_eda.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\nEDA completed successfully!")
print("Dashboard saved at:")
print(OUTPUT_PATH)

plt.show()