
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sales Prediction App", page_icon="📈", layout="centered")
st.title("📈 Sales Prediction App")
st.write("Enter your advertising budgets below to predict sales (in thousands)")


# Load trained model and scaler
try:
    model = joblib.load('sales_model.pkl')
    scaler = joblib.load('sales_scaler.pkl')
except Exception as e:
    st.error("Model or scaler not found. Please train and save them using your training script.")
    st.stop()

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
