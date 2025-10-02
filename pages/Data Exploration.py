

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

from theme import apply_theme, header

st.set_page_config(page_title="ExoLab • Data Exploration", page_icon="📊", layout="wide")

apply_theme(page_key="sage", nebula_path="assets/nebula.jpg")

header("📊 Data Exploration", "Visualize and analyze raw exoplanet datasets")

st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="exploration_csv")

st.markdown('<div class="glass"><h3>Exploration Dashboard</h3></div>', unsafe_allow_html=True)

