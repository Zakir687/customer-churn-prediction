from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------
# Paths
# ------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "models"
    / "best_churn_model.joblib"
)


# ------------------------------------------------
# Load Model
# ------------------------------------------------

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


model = load_model()


# ------------------------------------------------
# Main Heading
# ------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    """
    This application uses a machine-learning model to
    estimate whether a telecom customer is likely to
    **Churn** or **Stay**.
    """
)

st.divider()


# ------------------------------------------------
# Customer Information
# ------------------------------------------------

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)


# ================================
# COLUMN 1
# ================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen_text = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    senior_citizen = (
        1
        if senior_citizen_text == "Yes"
        else 0
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# ================================
# COLUMN 2
# ================================

with col2:

    if phone_service == "No":

        multiple_lines = "No phone service"

        st.info(
            "Multiple Lines: No phone service"
        )

    else:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes"]
        )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    if internet_service == "No":

        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"

        st.info(
            "Internet-dependent services are "
            "automatically set to No internet service."
        )

    else:

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes"]
        )


# ================================
# COLUMN 3
# ================================

with col3:

    if internet_service == "No":

        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

    else:

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes"]
        )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# ------------------------------------------------
# Charges
# ------------------------------------------------

st.subheader("Billing Information")

charge1, charge2 = st.columns(2)

with charge1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )


with charge2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


# ------------------------------------------------
# Prediction Button
# ------------------------------------------------

st.divider()

if st.button(
    "Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    # Create dataframe for prediction

    customer_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }
        ]
    )


    # Make prediction

    prediction = model.predict(
        customer_data
    )[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]


    # ------------------------------------------------
    # Display Results
    # ------------------------------------------------

    st.subheader("Prediction Result")

    result1, result2 = st.columns(2)

    with result1:

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to CHURN"
            )

        else:

            st.success(
                "✅ Customer is likely to STAY"
            )


    with result2:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    # ------------------------------------------------
    # Risk Level
    # ------------------------------------------------

    if probability >= 0.70:

        st.warning(
            "🔴 High Churn Risk"
        )

    elif probability >= 0.40:

        st.warning(
            "🟠 Medium Churn Risk"
        )

    else:

        st.success(
            "🟢 Low Churn Risk"
        )


    # ------------------------------------------------
    # Recommendation
    # ------------------------------------------------

    if probability >= 0.50:

        st.write(
            """
            **Recommended Action:**  
            Consider offering the customer a retention
            package, contract upgrade, discount, or
            additional support.
            """
        )

    else:

        st.write(
            """
            **Recommended Action:**  
            The customer currently has relatively low
            churn risk. Continue monitoring customer
            satisfaction and service quality.
            """
        )


# ------------------------------------------------
# Footer
# ------------------------------------------------

st.divider()

st.caption(
    "End-to-End Customer Churn Prediction "
    "and Deployment Project"
)