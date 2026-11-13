import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="NASA Fire Predictor", page_icon="🔥")
st.title("🔥 NASA Fire Safety Predictor")
st.write("Test how fire behaves in space vs Earth")

# Load model
model = joblib.load("model.pkl")

# 1. Inputs - all 7
st.sidebar.header("Test Conditions")

fuel_state = st.sidebar.selectbox("Fuel State", ["Solid", "Liquid", "Gas"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["PMMA", "Paper", "Polyethylene", "Wood", "Other"])
gravity = st.sidebar.slider("Gravity Level (g)", 0.0, 1.0, 1.0, step=0.01)
pressure = st.sidebar.slider("Ambient Pressure (kPa)", 10, 150, 101)
oxygen = st.sidebar.slider("Oxygen %", 10, 40, 21)
flow = st.sidebar.slider("Flow Velocity (m/s)", 0.0, 2.0, 0.1)
dim = st.sidebar.slider("Characteristic Size (mm)", 1, 100, 10)

if st.button("Predict Fire Risk"):
    # Create input dataframe - must match training order
    # We use simple encoding: Solid=0, Liquid=1, Gas=2
    fuel_state_map = {"Solid": 0, "Liquid": 1, "Gas": 2}
    
    # For Fuel_Type, we need to match your Colab encoding
    # If your model used OneHot, this simplified version uses label too
    # Most common in your project: PMMA=0, Paper=1, etc.
    fuel_type_map = {"PMMA": 0, "Paper": 1, "Polyethylene": 2, "Wood": 3, "Other": 4}

    input_data = pd.DataFrame([[
        fuel_state_map[fuel_state],
        fuel_type_map[fuel_type],
        gravity,
        pressure,
        oxygen,
        flow,
        dim
    ]], columns=['Fuel_State', 'Fuel_Type', 'Gravity_Level_g', 'Ambient_Pressure_kPa', 'Oxygen_Conc_pct', 'Flow_Velocity_m_s', 'Characteristic_Dim_mm'])

    prediction = model.predict(input_data)
    
    st.success(f"Prediction: {prediction[0]}")
    st.write("Try: Gravity = 0 + Oxygen = 30% = Very dangerous in space!")
