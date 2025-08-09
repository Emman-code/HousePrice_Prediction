# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model & columns
model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 Paris House Price Prediction")
st.write("This app predicts house prices based on your input features.")

# User input form
input_data = {}
for col in model_columns:
    input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: €{prediction:.2f}")
