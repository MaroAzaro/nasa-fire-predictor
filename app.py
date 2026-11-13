
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('/content/drive/MyDrive/NASA/nasa_fire_model.pkl')

st.title("NASA Microgravity Fire Predictor 🔥")
st.write("Will the fire go out in space?")

fuel = st.selectbox("Fuel Type", ['Nomex_Fabric', 'Methane', 'SIBAL_Cotton_Fiberglass', 'PMMA', 'Cellulose'])
state = st.selectbox("Fuel State", ['Solid', 'Gas', 'Liquid_Droplet', 'Solid_LargeScale'])
campaign = st.selectbox("Campaign", ['BASS-II', 'SAFFIRE', 'ACME', 'FLEX'])
oxygen = st.slider("Oxygen %", 14, 35, 21)
pressure = st.slider("Pressure kPa", 50, 200, 101)
flow = st.slider("Flow Velocity m/s", 0.0, 0.8, 0.2)
size = st.slider("Fuel Size mm", 0.2, 10.0, 2.0)

if st.button("Predict"):
    data = pd.DataFrame([{
        'Fuel_Type': fuel,
        'Fuel_State': state,
        'Experiment_Campaign': campaign,
        'Gravity_Level_g': 0.000001,
        'Ambient_Pressure_kPa': pressure,
        'Oxygen_Conc_pct': oxygen,
        'Flow_Velocity_m_s': flow,
        'Characteristic_Dim_mm': size
    }])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]*100
    
    if pred == 1:
        st.error(f"WILL GO OUT - {prob:.1f}% chance of extinction")
    else:
        st.success(f"WILL KEEP BURNING - Only {prob:.1f}% chance of going out")
