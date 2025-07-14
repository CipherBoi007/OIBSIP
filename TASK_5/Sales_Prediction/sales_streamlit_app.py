import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Sales Prediction App", page_icon="📈", layout="centered")
st.title("📈 Sales Prediction App")
st.write("Enter your advertising budgets below to predict sales (in thousands)")

# Load and train model (for demo, retrain each time; for production, use joblib to load)
data = pd.read_csv('Advertising.csv').iloc[:, 1:]
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_scaled, y)

# User input
st.header("Input Advertising Budgets")
tv = st.number_input("TV Budget (in thousands)", min_value=0.0, value=100.0, step=1.0)
radio = st.number_input("Radio Budget (in thousands)", min_value=0.0, value=25.0, step=1.0)
newspaper = st.number_input("Newspaper Budget (in thousands)", min_value=0.0, value=20.0, step=1.0)

if st.button("Predict Sales"):
    user_input = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })
    user_input_scaled = scaler.transform(user_input)
    predicted_sales = model.predict(user_input_scaled)[0]
    st.success(f"Predicted Sales: {predicted_sales:.2f} (in thousands)")

st.markdown("---")
st.subheader("Model Info")
st.write("SVR (Support Vector Regression) with RBF kernel, StandardScaler preprocessing.")
