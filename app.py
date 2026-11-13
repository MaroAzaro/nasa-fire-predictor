import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="NASA Fire Predictor", page_icon="🔥")
st.title("🔥 NASA Fire Safety Predictor")
st.write("Test how fire behaves in space vs Earth - Real NASA Data")

model = joblib.load("model.pkl")

st.sidebar.header("Test Conditions - All 8 Inputs")

fuel_state = st.sidebar.selectbox("Fuel State", ["Solid", "Liquid", "Gas"])
fuel_type = st.sidebar.selectbox("Fuel Type", ['Methane', 'Nomex_Fabric', 'Nomex_Spacecraft_Liner', 'Propane', 'PMMA', 'n_Heptane', 'SIBAL_Cotton_Fiberglass', 'Methanol', 'n_Decane', 'PMMA_Large_Slab', 'Dodecane_Hexadecane_Mix', 'Cotton_Fiberglass_Blend', 'Cellulose_Sheet', 'Hydrogen_Methane_Blend', 'Ethylene', 'Silicone_Composite'])
campaign = st.sidebar.selectbox("Experiment Campaign", ['ACME', 'BASS-II', 'SAFFIRE', 'FLEX'])
gravity = st.sidebar.slider("Gravity Level (g)", 0.0, 1.0, 1.0, step=0.01)
pressure = st.sidebar.slider("Ambient Pressure (kPa)", 10, 150, 101)
oxygen = st.sidebar.slider("Oxygen %", 10, 40, 21)
flow = st.sidebar.slider("Flow Velocity (m/s)", 0.0, 2.0, 0.1)
dim = st.sidebar.slider("Characteristic Size (mm)", 1, 100, 10)

if st.button("Predict Fire Risk"):
    input_data = pd.DataFrame([{
        'Fuel_State': fuel_state,
        'Fuel_Type': fuel_type,
        'Experiment_Campaign': campaign,
        'Gravity_Level_g': gravity,
        'Ambient_Pressure_kPa': pressure,
        'Oxygen_Conc_pct': oxygen,
        'Flow_Velocity_m_s': flow,
        'Characteristic_Dim_mm': dim
    }])

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]

    if prediction[0] == 0:
        st.error("🔥 DANGER - Will Ignite / Burning!")
    else:
        st.success("✅ SAFE - Flame Extinguished")

    st.write(f"Confidence: {max(proba)*100:.1f}%")
    st.info("Try: Campaign = SAFFIRE + Gravity = 0 + Oxygen = 30%")

# --- FOOTER ---
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 13px; padding-top: 10px;">
        <b>🔥 NASA Fire Safety Predictor</b><br>
        Built with Real NASA Microgravity Fire Data | ACME • BASS-II • SAFFIRE • FLEX<br>
        © 2026 NASA Fire Predictor. All Rights Reserved.<br>
        Made for Science & Space Safety | Version 1.0<br>
        <span style="font-size: 11px;">Disclaimer: For educational / demo purposes only. Not for real spacecraft safety decisions.</span>
    </div>
    """,
    unsafe_allow_html=True
)
