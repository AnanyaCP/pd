# app.py

import streamlit as st
import pandas as pd
from model import train_model

# -------------------------------
# Load model
# -------------------------------
model, feature_cols = train_model()

# -------------------------------
# UI
# -------------------------------
st.title("📊 Product Rating Predictor")

st.write("Enter product details:")

# Inputs
price = st.number_input("Price")
discount = st.number_input("Discount")
stock = st.selectbox("Stock", [0, 1])
category = st.selectbox("Category", ["A", "B", "C", "D", "UNKNOWN"])

# -------------------------------
# Prepare input
# -------------------------------
input_data = {
    "Price": price,
    "Discount": discount,
    "Stock": stock,
    "Category_A": 0,
    "Category_B": 0,
    "Category_C": 0,
    "Category_D": 0,
    "Category_UNKNOWN": 0
}

# Set selected category
input_data[f"Category_{category}"] = 1

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # 🔥 IMPORTANT FIX (match training columns)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Predict
    result = model.predict(input_df)

    st.success(f"Predicted Rating: {result[0]:.2f}")
