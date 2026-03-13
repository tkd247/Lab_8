import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load artifacts
model = tf.keras.models.load_model("artifacts/housing_model.h5")
scaler = joblib.load("artifacts/scaler.pkl")

st.title("Hamilton County Housing Value Predictor")
st.caption("Educational use only. Predictions are approximate.")

# Inputs
acres = st.number_input(
    "Land area (acres)",
    min_value=0.01,
    max_value=20.0,
    value=0.25,
    step=0.01
)

yearbuilt = st.number_input(
    "Year built",
    min_value=1900,
    max_value=2026,
    value=2000
)

sizearea = st.number_input(
    "Building area (sq ft)",
    min_value=300,
    max_value=10000,
    value=1800,
    step=50
)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "CALC_ACRES":[acres],
        "YEARBUILT":[yearbuilt],
        "SIZEAREA":[sizearea]
    })

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0][0]

    st.success(f"Estimated appraised value: ${prediction:,.0f}")
