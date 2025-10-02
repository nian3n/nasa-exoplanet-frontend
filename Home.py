import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoLab • Home", page_icon="🪐", layout="wide")

# Pick a themed color + nebula header image
apply_theme(page_key="deep_teal", nebula_path="assets/Nebula.png")

# Nebula hero with title + subtitle
header("🪐 ExoLab ", "Best ML tool to explore exoplanets and astroworld")

# Content
st.markdown(
    '<div class="glass"><h3>Welcome</h3><p>Exolab is aimed to </p></div>',
    unsafe_allow_html=True
)


