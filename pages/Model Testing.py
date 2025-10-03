import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Model Testing", page_icon="📈", layout="wide")

apply_theme(page_key="forest", nebula_path="assets/nebula.jpg")

header("📈 Model Testing", "Upload your exoplanet datasets and see how our model performs")

st.file_uploader("Upload evaluation dataset (CSV)", type=["csv"], key="evaluation_csv")

st.markdown('<div class="glass"><h3>Evaluation Results</h3></div>', unsafe_allow_html=True)
