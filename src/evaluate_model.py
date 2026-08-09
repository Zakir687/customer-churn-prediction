from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)


# ------------------------------------------------
# 1. Paths
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "cleaned_churn_data.csv"

MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"

REPORTS_PATH = ROOT / "reports"

OUTPUT_PATH = REPORTS_PATH / "model_performance_dashboard.png"


# ------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ------------------------------------------------
# 3. Prepare Features and Target
# ------------------------------------------------

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ------------------------------------------------
# 4. Recreate Same Train-Test Split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# 5. Load Best Model
# ------------------------------------------------

model = joblib.load(MODEL_PATH)

print("\nBest model loaded successfully!")


# ------------------------------------------------
# 6. Predictions
# ------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------
# 7. Calculate ROC
# ------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)


# ------------------------------------------------
# 8. Precision-Recall Curve
# ------------------------------------------------

precision, recall, thresholds_pr = precision_recall_curve(
    y_test,
    y_probability
)


# ------------------------------------------------
# 9. Confusion Matrix
# ------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


# ------------------------------------------------
# 10. Feature Importance
# ------------------------------------------------

preprocessor = model.named_steps["preprocessing"]

classifier = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

importance = classifier.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)


# Clean feature names

feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("numerical__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)


# ------------------------------------------------
# 11. Create ONE Dashboard
# ------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


# ------------------------------------------------
# Plot 1: Confusion Matrix
# ------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)

disp.plot(
    ax=axes[0, 0],
    values_format="d"
)

axes[0, 0].set_title(
    "Confusion Matrix"
)


# ------------------------------------------------
# Plot 2: ROC Curve
# ------------------------------------------------

axes[0, 1].plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc_score:.3f}"
)

axes[0, 1].plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

axes[0, 1].set_title(
    "ROC Curve"
)

axes[0, 1].set_xlabel(
    "False Positive Rate"
)

axes[0, 1].set_ylabel(
    "True Positive Rate"
)

axes[0, 1].legend()


# ------------------------------------------------
# Plot 3: Precision-Recall Curve
# ------------------------------------------------

axes[1, 0].plot(
    recall,
    precision
)

axes[1, 0].set_title(
    "Precision-Recall Curve"
)

axes[1, 0].set_xlabel(
    "Recall"
)

axes[1, 0].set_ylabel(
    "Precision"
)


# ------------------------------------------------
# Plot 4: Feature Importance
# ------------------------------------------------

axes[1, 1].barh(
    feature_importance["Feature"][::-1],
    feature_importance["Importance"][::-1]
)

axes[1, 1].set_title(
    "Top 10 Important Features"
)

axes[1, 1].set_xlabel(
    "Feature Importance"
)


# ------------------------------------------------
# Main Title
# ------------------------------------------------

fig.suptitle(
    "Gradient Boosting Churn Model Performance",
    fontsize=18
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)


# ------------------------------------------------
# Save Dashboard
# ------------------------------------------------

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\nPerformance dashboard saved successfully!")
print(OUTPUT_PATH)

plt.show()