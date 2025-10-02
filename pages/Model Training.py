import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoLab • Model Training", page_icon="🤖", layout="wide")

apply_theme(page_key="ochre", nebula_path="assets/Nebula.png")

header("🤖 Model Training", "Upload training data and build models")

st.file_uploader("Upload training data (CSV)", type=["csv"], key="training_csv")

st.markdown('<div class="glass"><h3>Training Interface</h3></div>', unsafe_allow_html=True)
