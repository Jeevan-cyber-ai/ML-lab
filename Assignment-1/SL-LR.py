import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess

# ----------------------------
# LOAD DATA
# ----------------------------
df = load_and_preprocess()

# ----------------------------
# DARK THEME
# ----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
.stSidebar {
    background-color: #161B22;
}
div.stButton > button {
    background-color: #1F2937;
    color: white;
    border-radius: 12px;
    border: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LINEAR REGRESSION
# ----------------------------
X_linear = df.drop(["battery_health_percent","replace_battery"], axis=1)
y_linear = df["battery_health_percent"]

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train_l, y_train_l)

# Save feature order
feature_columns = X_linear.columns

# ----------------------------
# LOGISTIC REGRESSION
# ----------------------------
X_log = df.drop(["replace_battery","battery_health_percent"], axis=1)
y_log = df["replace_battery"]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_log, y_train_log)

# ----------------------------
# UI
# ----------------------------
st.title("💻 Laptop Battery Health & Replacement Predictor")
st.header("Enter Laptop Details")

model_year = st.number_input("Model Year", min_value=2000, max_value=2030)
daily_usage_hours = st.slider("Daily Usage Hours", 1, 15)
charging_cycles = st.number_input("Charging Cycles")
avg_charge_limit_percent = st.slider("Avg Charge Limit Percent", 50, 100)
battery_age_months = st.number_input("Battery Age (Months)")
overheating_issues = st.selectbox("Overheating Issues", [0, 1])
performance_rating = st.slider("Performance Rating", 1, 5)

brand = st.selectbox("Brand", ["Apple", "Asus", "Dell", "HP", "Lenovo"])
os_type = st.selectbox("Operating System", ["Windows", "macOS"])
usage_type = st.selectbox("Usage Type", ["Gaming", "Office", "Programming", "Student"])

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict"):

    # Create dictionary for input
    input_dict = {
        "model_year": model_year,
        "daily_usage_hours": daily_usage_hours,
        "charging_cycles": charging_cycles,
        "avg_charge_limit_percent": avg_charge_limit_percent,
        "battery_age_months": battery_age_months,
        "overheating_issues": overheating_issues,
        "performance_rating": performance_rating,
        "brand_Asus": 1 if brand == "Asus" else 0,
        "brand_Dell": 1 if brand == "Dell" else 0,
        "brand_HP": 1 if brand == "HP" else 0,
        "brand_Lenovo": 1 if brand == "Lenovo" else 0,
        "os_Windows": 1 if os_type == "Windows" else 0,
        "usage_type_Office": 1 if usage_type == "Office" else 0,
        "usage_type_Programming": 1 if usage_type == "Programming" else 0,
        "usage_type_Student": 1 if usage_type == "Student" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order & fill missing columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # ----------------------------
    # LINEAR PREDICTION
    # ----------------------------
    health_prediction = linear_model.predict(input_df)
    st.success(f"Predicted Battery Health: {round(health_prediction[0],2)} %")

    # ----------------------------
    # LOGISTIC PREDICTION
    # ----------------------------
    replace_prediction = log_model.predict(input_df)

    if replace_prediction[0] == 1:
        st.error("⚠️ Battery Replacement Recommended")
    else:
        st.success("✅ Battery is in Good Condition")