
import streamlit as st
import pandas as pd
import pickle
from datetime import date

# ==========================================
# Page Configuration

st.set_page_config(
    page_title="Healthy Meals Renewal Prediction",
    page_icon="🥗",
    layout="centered"
)

# ==========================================
# Load model and encoder

with open("churn_rf_healthy_meals_all_features.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("churn_encoder_healthy_meals_all_features.pkl", "rb") as f:
    encoder = pickle.load(f)

# ==========================================
# Title

st.title("🥗 Healthy Meals Renewal Prediction")
st.write("Enter customer information to predict renewal probability.")

# ==========================================
# Demographic Features

st.subheader("Demographic Features")

col1, col2 = st.columns(2)
with col1:

    age = st.number_input(
        "Age (20-100)",
        min_value=20,
        max_value=100,
        value=35
    )
    
    income_level = st.selectbox(
        "Income Level",
        [
            "High",
            "Low",
            "Medium",
            "Very High"
        ]
    )

with col2:

    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "High School",
            "Other",
            "Post-Graduate"
        ]
    )

    tech_comfort_score = st.number_input(
        "Tech Comfort Score (1-5)",
        min_value=1,
        max_value=5,
        value=3
    )

# ==========================================
# Device Information

st.subheader("Device Information")


device_type = st.selectbox(
    "Device Type",
    [
        "Desktop-only",
        "Mobile-only",
        "Multi-device"
    ]
)

# ==========================================
# Usage Features

st.subheader("Usage Features")

st.info(
    "Activity features should be obtained from customer usage records."
)

col1, col2 = st.columns(2)

with col1:

    total_sessions = st.number_input(
        "Total Sessions (0-15,601)",
        min_value=0,
        max_value=15601,
        value=50
    )

    active_days = st.number_input(
        "Active Days (0-365)",
        min_value=0,
        max_value=365,
        value=30
    )

    h2_sessions = st.number_input(
        "H2 Sessions (Jul-Dec) (0-122)",
        min_value=0,
        max_value=122,
        value=25
    )

with col2:

    gross_session_length = st.number_input(
        "Gross Session Length (0-15,601)",
        min_value=0.0,
        max_value=15601.0,
        value=500.0
    )
    
    active_quarters = st.number_input(
        "Active Quarters (0-4)",
        min_value=0,
        max_value=4,
        value=3
    )

    last_activity_date = st.date_input(
        "Last Activity Date",
        min_value=date(2024,1,1),
        max_value=date(2025,1,1),
        value=date(2024,12,20)
    )

# ==========================================
# Prediction

if st.button(
    "Predict Renewal Probability",
    use_container_width=True
):

    # --------------------------------------
    # Feature Engineering

    avg_sessions_per_active_quarter = (
        total_sessions / active_quarters
        if active_quarters > 0
        else 0
    )

    avg_session_length = (
        gross_session_length / total_sessions
        if total_sessions > 0
        else 0
    )

    sessions_per_active_day = (
        total_sessions / active_days
        if active_days > 0
        else 0
    )

    quarter_activity_ratio = (
        active_quarters / 4
    )

    h1_sessions = (
        total_sessions - h2_sessions
    )

    h2_h1_session_ratio = (
        h2_sessions / h1_sessions
        if h1_sessions > 0
        else 0
    )

    reference_date = date(2025,1,1)

    days_since_last_activity = (
        reference_date - last_activity_date
    ).days

    # --------------------------------------
    # Create Input DataFrame

    input_df = pd.DataFrame({

        "TOTAL_SESSIONS": [total_sessions],
        "GROSS_SESSION_LENGTH": [gross_session_length],
        "ACTIVE_DAYS": [active_days],
        "ACTIVE_QUARTERS": [active_quarters],
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": [avg_sessions_per_active_quarter],
        "H2_SESSIONS": [h2_sessions],
        "H2_H1_SESSION_RATIO": [h2_h1_session_ratio],
        "AVG_SESSION_LENGTH": [avg_session_length],
        "SESSIONS_PER_ACTIVE_DAY": [sessions_per_active_day],
        "QUARTER_ACTIVITY_RATIO": [quarter_activity_ratio],
        "DAYS_SINCE_LAST_ACTIVITY": [days_since_last_activity],
        "AGE": [age],
        "TECH_COMFORT_SCORE": [tech_comfort_score],
        "INCOME_LEVEL": [income_level],
        "EDUCATION": [education],
        "DEVICE_TYPE": [device_type]

    })

    # --------------------------------------
    # Encoding

    categorical_cols = [
        "INCOME_LEVEL",
        "EDUCATION",
        "DEVICE_TYPE"
    ]

    encoded = encoder.transform(
        input_df[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_cols
        )
    )

    final_input = pd.concat(
        [
            input_df.drop(
                columns=categorical_cols
            ).reset_index(drop=True),

            encoded_df.reset_index(drop=True)

        ],
        axis=1
    )

    # Ensure same feature order

    final_input = final_input[
        model.feature_names_in_
    ]

    # --------------------------------------
    # Prediction

    probabilities = model.predict_proba(
        final_input
    )[0]

    renewal_probability = probabilities[1]

    churn_probability = (
        1 - renewal_probability
    )

    # --------------------------------------
    # Result

    st.divider()

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Renewal Probability",
            f"{renewal_probability:.1%}"
        )

    with col2:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    if renewal_probability >= 0.5:

        st.success(
            "✅ Churn Risk: Low"
        )

    else:

        st.warning(
            "⚠️ Churn Risk: High"
        )
