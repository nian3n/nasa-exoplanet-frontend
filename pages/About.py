import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • About", page_icon="🛰️", layout="wide")

# Use a distinct page color from your palette (e.g., "royal")
apply_theme(page_key="sand", nebula_path="assets/nebula.jpg")

header("About ExoHorizon", "Team & Project Overview")

DARK_BROWN = "#DDE2EE"
WHITE = "#FFFFFF"
st.markdown(
    f"""
    <style>
      .glass h3,p,li {{
        color: {DARK_BROWN} !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

#Team
st.markdown(
    """
    <div class="glass">
      <h3>Our Team</h3>
      <p>We are a bunch of aerospace amateurs from all over the world and we built ExoHorizon, a practical machine-learning tool for interfacing with exoplanet data.</p>
      <ul>
        <li><b>Olivia Song</b></li>
        <li><b>Matias Freire</b></li>
        <li><b>Wisam Kakooz</b></li>
        <li><b>F. Yalın Şen</b></li>
        <li><b>Farheen Rahman</b></li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Project intro ---
st.markdown(
    """
    <div class="glass">
      <h3>What is ExoHorizon?</h3>
      <p>
        ExoHorizon is our project for NASA SpaceApps 2025 challenge: an end-to-end workflow to ingest exoplanet datasets, 
        engineer features, and train/evaluate ML models that classify candidates. 
        It’s designed for both researchers who want a quick, reproducible baseline and 
        novices who want an approachable interface to explore light curves and catalog features.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

#link to project demo
st.markdown(
    """
    <div class="glass">
      <h3>Project Demo</h3>
      <ul>
        <li>Link to Project Demo</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
