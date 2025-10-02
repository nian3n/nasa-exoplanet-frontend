
import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoLab • About", page_icon="🛰️", layout="wide")

# Use a distinct page color from your palette (e.g., "royal")
apply_theme(page_key="sand", nebula_path="assets/Nebula.png")

header("🛰️ About ExoLab", "Team & Project Overview")

DARK_BROWN = "#4A2F14"
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

# --- Team ---
st.markdown(
    """
    <div class="glass">
      <h3>Our Team</h3>
      <p>We are aerospace amateurs from around the world and we built a practical machine-learning tool for interacting with exoplanet data.</p>
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
      <h3>What is ExoLab?</h3>
      <p>
        ExoLab is our project for NASA SpaceApps 2025 challenge: an end-to-end workflow to ingest exoplanet survey data, 
        engineer features, and train/evaluate ML models that classify candidates. 
        It’s designed for both researchers who want a quick, reproducible baseline and 
        novices who want an approachable interface to explore light curves and catalog features.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Modeling plan ---
st.markdown(
    """
    <div class="glass">
      <h3>Modeling Stack & Pipeline</h3>
      <p>Python: Pandas · NumPy · scikit-learn (with options for LightGBM / XGBoost )</p>
      <ol>
        <li><b>Identify Important Features → Feature Engineering</b><br/>
            Domain-informed transforms (e.g., light-curve statistics, period/frequency features, summary aggregations).
        </li>
        <li><b>Encode Categorical Columns</b><br/>
            One-Hot Encoders / (Multi)Label encoders as needed for catalog flags & discrete fields.
        </li>
        <li><b>Pipelines</b><br/>
            Normalize/scale where appropriate, impute missing values, and keep transformations reproducible.
        </li>
        <li><b>Model Candidates</b><br/>
            <ul>
              <li><b>LightGBM</b> — fast, strong baseline for tabular data (primary)</li>
              <li><b>XGBoost</b> — alternative gradient boosting</li>
              <li><b>Random Forest</b> — simple baseline (likely weaker, but useful for comparison)</li>
            </ul>
        </li>
        <li><b>Train / Test Split & Metrics</b><br/>
            Accuracy, Precision/Recall, F1, ROC-AUC; confusion matrix for class balance insights.
        </li>
        <li><b>Hyperparameters</b><br/>
            UI controls to tweak key params (learning rate, estimators, depth, regularization). 
            <i>(Challenging but planned via Streamlit widgets.)</i>
        </li>
      </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Web interface plan ---
st.markdown(
    """
    <div class="glass">
      <h3>Web Interface</h3>
      <p><b>Stack:</b> Streamlit (or HTML/CSS/JS) with Nebula-themed UI</p>
      <ul>
        <li><b>Facilitate user interaction</b> — guided upload, schema checks, and pipeline presets.</li>
        <li><b>Show statistics</b> — live accuracy/F1/ROC-AUC, learning curves, and confusion matrices.</li>
        <li><b>Ingest new data</b> — append to datasets and retrain models on demand.</li>
        <li><b>Hyperparameter controls</b> — sliders/selects for key model knobs.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Optional: quick CTA or links (fill in later if you add docs/demo)
st.markdown(
    """
    <div class="glass">
      <h3>Roadmap</h3>
      <ul>
        <li>Data loaders & schema validation</li>
        <li>Feature engineering presets</li>
        <li>Baseline LightGBM training + evaluation dashboard</li>
        <li>Hyperparameter UI + experiment tracking</li>
        <li>Model export & inference on new uploads</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
