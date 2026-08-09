from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ------------------------------------------------
# 1. Paths
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "cleaned_churn_data.csv"

REPORT_PATH = ROOT / "reports" / "model_comparison.csv"

BEST_MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"


# ------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ------------------------------------------------
# 3. Features and Target
# ------------------------------------------------

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ------------------------------------------------
# 4. Numerical and Categorical Features
# ------------------------------------------------

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column
    for column in X.columns
    if column not in numerical_features
]


# ------------------------------------------------
# 5. Preprocessing
# ------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ------------------------------------------------
# 6. Train-Test Split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# 7. Define Models
# ------------------------------------------------

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ),

    "Gradient Boosting":

        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42
        )
}


# ------------------------------------------------
# 8. Train and Evaluate Models
# ------------------------------------------------

results = []

trained_models = {}

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")


# ------------------------------------------------
# 9. Model Comparison
# ------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------
# 10. Save Results
# ------------------------------------------------

REPORT_PATH.parent.mkdir(
    exist_ok=True
)

results_df.to_csv(
    REPORT_PATH,
    index=False
)

print("\nComparison saved at:")
print(REPORT_PATH)


# ------------------------------------------------
# 11. Select Best Model
# ------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\nBest Model:")
print(best_model_name)


# ------------------------------------------------
# 12. Save Best Model
# ------------------------------------------------

joblib.dump(
    best_model,
    BEST_MODEL_PATH
)

print("\nBest model saved successfully!")
print(BEST_MODEL_PATH)