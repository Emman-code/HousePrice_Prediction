# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model & feature list
model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 Paris House Price Prediction")
st.write("Enter a few details to predict the house price.")

# User input
input_data = {}
for col in model_columns:
    if col in ["hasYard", "hasPool"]:  # Binary input
        input_data[col] = st.selectbox(f"{col} (0 = No, 1 = Yes)", [0, 1])
    else:  # Numeric input
        input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: €{prediction:.2f}")

