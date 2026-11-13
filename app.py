import streamlit as st
import sys, sklearn, numpy, joblib

st.write("Python:", sys.version)
st.write("sklearn:", sklearn.__version__)
st.write("numpy:", numpy.__version__)
st.write("joblib:", joblib.__version__)
st.stop()  # stop here, don't load model yet
