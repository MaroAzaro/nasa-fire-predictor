import streamlit as st
import joblib
import pandas as pd
import base64

st.set_page_config(page_title="NASA Fire Predictor", page_icon="🔥")

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("background.png")

st.markdown(f"""
<style>
:root {{
    color-scheme: dark !important;
}}
html, body {{
    background-color: #020617 !important;
    color: white !important;
}}
[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {{
    background: transparent !important;
    color: white !important;
}}
.stApp {{
    background-image: url("data:image/png;base64,{bg}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    background-color: #020617 !important;
    color: white !important;
}}
@media (prefers-color-scheme: light) {{
    :root {{ color-scheme: dark !important; }}
    html, body {{
        background-color: #020617 !important;
        color: white !important;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent !important;
        color: white !important;
    }}
    .stApp {{
        background-image: url("data:image/png;base64,{bg}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-color: #020617 !important;
        color: white !important;
    }}
    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: white !important;
    }}
}}
h1, h2, h3, p, span, label, div {{
    color: white !important;
}}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
    background-color: #0a1128 !important;
    color: white !important;
}}
[data-testid="stSidebar"] * {{
    color: white !important;
}}
.stButton > button, [data-testid="stSelectbox"] > div > div, [data-testid="stSlider"] > div {{
    background-color: #1e293b !important;
    color: white !important;
    border-color: #334155 !important;
}}
input, select, textarea {{
    background-color: #1e293b !important;
    color: white !important;
}}
@media (max-width: 768px) {{
    .stApp {{
        background-image: url("data:image/png;base64,{bg}") !important;
        background-size: 90% 90% !important;
        background-position: center center !important;
        background-attachment: scroll !important;
        background-repeat: no-repeat !important;
        background-color: #020617 !important;
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 80px !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

st.title("🔥 NASA Fire Safety Predictor")
st.write("Test how fire behaves in space vs Earth - Real NASA Data")

model = joblib.load("model.pkl")

# --- CENTERED LOGO IN SIDEBAR ---
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image("logo.png", width=240)


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
