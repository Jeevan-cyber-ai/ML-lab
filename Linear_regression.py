import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Linear Regression Visualization (Sklearn Dataset)")

# Load dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

st.subheader("Dataset Preview")
st.write(X.head())

# Select feature for simple regression
feature = st.selectbox("Select Feature for Regression", X.columns)

X_selected = X[[feature]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model parameters
st.subheader("Model Parameters")
st.write("Coefficient:", model.coef_[0])
st.write("Intercept:", model.intercept_)

# Evaluation Metrics
st.subheader("Evaluation Metrics")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("Mean Squared Error:", mse)
st.write("R² Score:", r2)

# Visualization
st.subheader("Actual vs Predicted")

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, label="Actual")
ax.plot(X_test, y_pred, label="Predicted")
ax.set_xlabel(feature)
ax.set_ylabel("Target")
ax.legend()

st.pyplot(fig)