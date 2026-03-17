import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
div.stButton > button {
    background-color: #1F2937;
    color: white;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Laptop Battery Clustering (KMeans)")

# ----------------------------
# PREPARE DATA FOR CLUSTERING
# ----------------------------

# Remove supervised targets
X_cluster = df.drop(["battery_health_percent", "replace_battery"], axis=1)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster column
df["cluster"] = kmeans.labels_

# ----------------------------
# SHOW CLUSTER DISTRIBUTION
# ----------------------------

st.header("Cluster Distribution")
st.write(df["cluster"].value_counts())

# ----------------------------
# SHOW CLUSTER CENTERS
# ----------------------------

centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X_cluster.columns
)

st.header("Cluster Centers (Average Feature Values)")
st.write(centers)

# ----------------------------
# VISUALIZATION
# ----------------------------

st.header("Cluster Visualization")

fig, ax = plt.subplots()

scatter = ax.scatter(
    df["battery_age_months"],
    df["battery_health_percent"],
    c=df["cluster"]
)

ax.set_xlabel("Battery Age (Months)")
ax.set_ylabel("Battery Health (%)")

st.pyplot(fig)

# ----------------------------
# USER INPUT FOR CLUSTER PREDICTION
# ----------------------------

st.header("Check Which Cluster Your Laptop Belongs To")

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

if st.button("Find Cluster"):

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

    input_df = pd.DataFrame([input_dict])

    # Match column order
    input_df = input_df.reindex(columns=X_cluster.columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict cluster
    cluster_pred = kmeans.predict(input_scaled)

    st.success(f"🔵 This laptop belongs to Cluster: {cluster_pred[0]}")