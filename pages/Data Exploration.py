import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px

from theme import apply_theme, header

#Page setup
st.set_page_config(page_title="ExoHorizon • Data Exploration", page_icon="📊", layout="wide")
apply_theme(page_key="sage", nebula_path="assets/nebula.jpg")
st.markdown("""
<style>
div[data-baseweb="popover"] { 
  color: #000 !important; 
  -webkit-text-fill-color: #000 !important; 
}

/* Options (normal) */
div[data-baseweb="popover"] [role="listbox"] [role="option"],
div[data-baseweb="popover"] [role="listbox"] [role="option"] *,
div[data-baseweb="menu-item"],
div[data-baseweb="menu-item"] * {
  color: #000 !important;
  -webkit-text-fill-color: #000 !important;
}

/* Selected / highlighted states (keep text black) */
div[role="listbox"] [aria-selected="true"],
div[role="listbox"] [aria-selected="true"] *,
div[role="listbox"] [data-highlighted="true"],
div[role="listbox"] [data-highlighted="true"] * {
  color: #000 !important;
  -webkit-text-fill-color: #000 !important;
}
</style>
""", unsafe_allow_html=True)

header("📊 Data Exploration", "Visualize and analyze raw exoplanet datasets")

#File Uploader 
uploaded = st.file_uploader("Upload your raw dataset", type=["csv"], key="exploration_csv")
# st.markdown('<div class="glass"><h3>Exploration Dashboard</h3></div>', unsafe_allow_html=True)  # commented out, replaced by image glass div below
# =====================
# IMAGE GLASS DIV BLOCK (add-on, easy to remove)
# =====================
import base64, os
def get_data_uri_img(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()

img_path = "nasa-exoplanet-frontend-main/assets/data_exploration_dashboard.jpg"
data_uri = get_data_uri_img(img_path)
if data_uri:
    st.markdown(
        f"""
        <div class='glass' style="background-image: url('{data_uri}'); background-size: cover; background-position: center; min-height: 200px; border-radius:1rem; overflow: hidden; margin-bottom:1.5rem;">
            <div style='backdrop-filter: blur(6px); padding:1.5rem; color:#fff;'>
                <h3 style='margin-top:0;'>Exploration Dashboard</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# =====================
# END IMAGE GLASS DIV BLOCK
# =====================

#Revelant KOI columns to keep for analysis
KOI_KEEP_COLS = [
    "koi_disposition", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_period", "koi_time0bk_err1", "koi_time0bk_err2", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_tce_plnt_num",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
    "koi_tce_delivname"  
]

#Must-exist KOI columns to identify a Kepler KOI dataset
KOI_ANCHOR_COLS = {"koi_disposition", "koi_period", "koi_prad", "koi_tce_delivname"}

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    import io
    # Read the raw bytes (Streamlit uploader is a file-like object)
    if hasattr(file, "seek"):
        file.seek(0)
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()

    # Decode to text
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)

    # Drop only the *leading* comment lines that start with '#'
    lines = text.splitlines()
    start = 0
    while start < len(lines) and lines[start].lstrip().startswith("#"):
        start += 1

    cleaned = "\n".join(lines[start:])
    return pd.read_csv(io.StringIO(cleaned))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names; create a mapping to access them uniformly."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = {c: c.lower() for c in df.columns}
    df.rename(columns=mapping, inplace=True)
    return df

def is_kepler_koi(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return KOI_ANCHOR_COLS.issubset(cols)

def subset_relevant_koi(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in KOI_KEEP_COLS if c in df.columns]
    return df[keep].copy()

def drop_false_positives_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop NaN koi_tce_delivname
    df = df[df["koi_tce_delivname"].notna()]
    # Drop False Positives (case-insensitive, handle weird spacing)
    if "koi_disposition" in df.columns:
        df["koi_disposition"] = df["koi_disposition"].astype(str).str.strip()
        df = df[~df["koi_disposition"].str.upper().eq("FALSE POSITIVE")]
    return df

# Main
if not uploaded:
    st.info("Upload a dataset (in CSV) to begin exploring your data.")
else:
    try:
        raw = load_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Normalize column names (lower-case keys)
    df = normalize_columns(raw)
    if not is_kepler_koi(df):
        st.error("Please upload a **Kepler KOI dataset**. The current tool only supports Kepler datasets.")
        st.stop()

    # Keep only relevant KOI columns
    df = subset_relevant_koi(df)

    # Clean: drop NaN in koi_tce_delivname, filter out False Positives
    df = drop_false_positives_and_nans(df)

    # Report basic shape
    st.success(f"Prepared KOI dataframe with **{df.shape[0]:,} rows** × **{df.shape[1]} columns**")

    # Make sure types are numeric where appropriate
    numeric_cols = [
        c for c in df.columns if c not in {"koi_disposition", "koi_tce_plnt_num", "koi_tce_delivname"}
    ]
    #Fill safe non-numeric conversions
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    #Chart picker
    st.subheader("Choose a visualization for your KOI dataset")
    st.markdown('<label class="exo-selectbox-label">Visualization type</label>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Visualization type",
        [
            "Period–Radius (scatter, log–log)",
            "Radius–Insolation (scatter, log X)",
            "Transit Duration vs Period (scatter)",
            "SNR vs Kepler Magnitude (scatter)",
            "Radius vs Equilibrium Temperature (scatter)",
            "Disposition counts (bar)",
            "Histogram: Period",
            "Histogram: Radius",
            "Histogram: Transit Depth",
            "Histogram: Transit Duration",
            "Histogram: SNR",
            "Histogram: Equilibrium Temperature",
            "Correlation Heatmap (numeric)"
        ],
        index=0,
        help="Select exactly one chart type. The chart will render below.",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Downsample slider for speed on very large tables
    row_limit = st.slider(
        "Choose the number of rows you would like to plot",
        min_value=1000, max_value=max(1000, len(df)), value=min(50000, len(df)), step=1000
    )
    sample = df.head(row_limit)

    # Helpers to guard missing columns
    def need(*cols):
        missing = [c for c in cols if c not in sample.columns]
        if missing:
            st.warning(f"Missing required columns for this chart: {', '.join(missing)}")
            return False
        if sample[list(cols)].dropna().empty:
            st.warning("Not enough non-NaN data to render this chart.")
            return False
        return True

    #Render the chosen chart
    if chart_type == "Period–Radius (scatter, log–log)":
        if need("koi_period", "koi_prad"):
            fig = px.scatter(
                sample, x="koi_period", y="koi_prad",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                hover_data=["koi_model_snr", "koi_kepmag"] if "koi_model_snr" in sample and "koi_kepmag" in sample else None,
                opacity=0.8, title="Period–Radius Diagram"
            )
            fig.update_xaxes(type="log", title="Orbital Period (days)")
            fig.update_yaxes(type="log", title="Planet Radius (Earth radii)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Radius–Insolation (scatter, log X)":
        if need("koi_insol", "koi_prad"):
            fig = px.scatter(
                sample, x="koi_insol", y="koi_prad",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Radius vs Insolation"
            )
            fig.update_xaxes(type="log", title="Insolation (S⊕)")
            fig.update_yaxes(title="Planet Radius (R⊕)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Transit Duration vs Period (scatter)":
        if need("koi_period", "koi_duration"):
            fig = px.scatter(
                sample, x="koi_period", y="koi_duration",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Transit Duration vs Period"
            )
            fig.update_xaxes(title="Orbital Period (days)")
            fig.update_yaxes(title="Transit Duration (hours)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "SNR vs Kepler Magnitude (scatter)":
        if need("koi_model_snr", "koi_kepmag"):
            fig = px.scatter(
                sample, x="koi_kepmag", y="koi_model_snr",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Detection SNR vs Kepler Magnitude"
            )
            fig.update_xaxes(title="Kepler Magnitude (Kp)")
            fig.update_yaxes(title="Model SNR")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Radius vs Equilibrium Temperature (scatter)":
        if need("koi_teq", "koi_prad"):
            fig = px.scatter(
                sample, x="koi_teq", y="koi_prad",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Radius vs Equilibrium Temperature"
            )
            fig.update_xaxes(title="Equilibrium Temperature (K)")
            fig.update_yaxes(title="Planet Radius (R⊕)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Disposition counts (bar)":
        if need("koi_disposition"):
            counts = (sample["koi_disposition"]
                      .astype("string").str.strip()
                      .value_counts(dropna=False)
                      .reset_index())
            counts.columns = ["koi_disposition", "count"]
            fig = px.bar(counts, x="koi_disposition", y="count", title="KOI Disposition Counts")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type.startswith("Histogram:"):
        # map choice to column
        col_map = {
            "Histogram: Period": "koi_period",
            "Histogram: Radius": "koi_prad",
            "Histogram: Transit Depth": "koi_depth",
            "Histogram: Transit Duration": "koi_duration",
            "Histogram: SNR": "koi_model_snr",
            "Histogram: Equilibrium Temperature": "koi_teq",
        }
        col = col_map[chart_type]
        if need(col):
            bins = st.slider("Bins", 5, 120, 40)
            fig = px.histogram(sample, x=col, nbins=bins, title=f"Histogram: {col}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Correlation Heatmap (numeric)":
        # Only numeric cols from our keep set
        num_keep = [c for c in KOI_KEEP_COLS if c in sample.columns and sample[c].dtype != "object"]
        if len(num_keep) < 2:
            st.warning("Need at least two numeric KOI columns for a correlation heatmap.")
        else:
            corr = sample[num_keep].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation (numeric KOI features)")
            st.plotly_chart(fig, use_container_width=True)

    # Expose the prepared dataframe for plotting (if you need elsewhere)
    st.markdown("#### Cleaned KOI dataset")
    st.dataframe(df, use_container_width=True)




#original
''''import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px

from theme import apply_theme, header

#Page setup
st.set_page_config(page_title="ExoHorizon • Data Exploration", page_icon="📊", layout="wide")
apply_theme(page_key="sage", nebula_path="assets/nebula.jpg")
st.markdown("""
<style>
div[data-baseweb="popover"] { 
  color: #000 !important; 
  -webkit-text-fill-color: #000 !important; 
}

/* Options (normal) */
div[data-baseweb="popover"] [role="listbox"] [role="option"],
div[data-baseweb="popover"] [role="listbox"] [role="option"] *,
div[data-baseweb="menu-item"],
div[data-baseweb="menu-item"] * {
  color: #000 !important;
  -webkit-text-fill-color: #000 !important;
}

/* Selected / highlighted states (keep text black) */
div[role="listbox"] [aria-selected="true"],
div[role="listbox"] [aria-selected="true"] *,
div[role="listbox"] [data-highlighted="true"],
div[role="listbox"] [data-highlighted="true"] * {
  color: #000 !important;
  -webkit-text-fill-color: #000 !important;
}
</style>
""", unsafe_allow_html=True)

header("📊 Data Exploration", "Visualize and analyze raw exoplanet datasets")

#File Uploader 
uploaded = st.file_uploader("Upload your raw dataset", type=["csv"], key="exploration_csv")
st.markdown('<div class="glass"><h3>Exploration Dashboard</h3></div>', unsafe_allow_html=True)

#Revelant KOI columns to keep for analysis
KOI_KEEP_COLS = [
    "koi_disposition", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_period", "koi_time0bk_err1", "koi_time0bk_err2", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_tce_plnt_num",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
    "koi_tce_delivname"  
]

#Must-exist KOI columns to identify a Kepler KOI dataset
KOI_ANCHOR_COLS = {"koi_disposition", "koi_period", "koi_prad", "koi_tce_delivname"}

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    import io
    # Read the raw bytes (Streamlit uploader is a file-like object)
    if hasattr(file, "seek"):
        file.seek(0)
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()

    # Decode to text
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)

    # Drop only the *leading* comment lines that start with '#'
    lines = text.splitlines()
    start = 0
    while start < len(lines) and lines[start].lstrip().startswith("#"):
        start += 1

    cleaned = "\n".join(lines[start:])
    return pd.read_csv(io.StringIO(cleaned))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names; create a mapping to access them uniformly."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = {c: c.lower() for c in df.columns}
    df.rename(columns=mapping, inplace=True)
    return df

def is_kepler_koi(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return KOI_ANCHOR_COLS.issubset(cols)

def subset_relevant_koi(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in KOI_KEEP_COLS if c in df.columns]
    return df[keep].copy()

def drop_false_positives_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop NaN koi_tce_delivname
    df = df[df["koi_tce_delivname"].notna()]
    # Drop False Positives (case-insensitive, handle weird spacing)
    if "koi_disposition" in df.columns:
        df["koi_disposition"] = df["koi_disposition"].astype(str).str.strip()
        df = df[~df["koi_disposition"].str.upper().eq("FALSE POSITIVE")]
    return df

# Main
if not uploaded:
    st.info("Upload a dataset (in CSV) to begin exploring your data.")
else:
    try:
        raw = load_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Normalize column names (lower-case keys)
    df = normalize_columns(raw)
    if not is_kepler_koi(df):
        st.error("Please upload a **Kepler KOI dataset**. The current tool only supports Kepler datasets.")
        st.stop()

    # Keep only relevant KOI columns
    df = subset_relevant_koi(df)

    # Clean: drop NaN in koi_tce_delivname, filter out False Positives
    df = drop_false_positives_and_nans(df)

    # Report basic shape
    st.success(f"Prepared KOI dataframe with **{df.shape[0]:,} rows** × **{df.shape[1]} columns**")

    # Make sure types are numeric where appropriate
    numeric_cols = [
        c for c in df.columns if c not in {"koi_disposition", "koi_tce_plnt_num", "koi_tce_delivname"}
    ]
    #Fill safe non-numeric conversions
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    #Chart picker
    st.subheader("Choose a visualization for your KOI dataset")
    st.markdown('<label class="exo-selectbox-label">Visualization type</label>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Visualization type",
        [
            "Period–Radius (scatter, log–log)",
            "Radius–Insolation (scatter, log X)",
            "Transit Duration vs Period (scatter)",
            "SNR vs Kepler Magnitude (scatter)",
            "Radius vs Equilibrium Temperature (scatter)",
            "Disposition counts (bar)",
            "Histogram: Period",
            "Histogram: Radius",
            "Histogram: Transit Depth",
            "Histogram: Transit Duration",
            "Histogram: SNR",
            "Histogram: Equilibrium Temperature",
            "Correlation Heatmap (numeric)"
        ],
        index=0,
        help="Select exactly one chart type. The chart will render below.",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Downsample slider for speed on very large tables
    row_limit = st.slider(
        "Choose the number of rows you would like to plot",
        min_value=1000, max_value=max(1000, len(df)), value=min(50000, len(df)), step=1000
    )
    sample = df.head(row_limit)

    # Helpers to guard missing columns
    def need(*cols):
        missing = [c for c in cols if c not in sample.columns]
        if missing:
            st.warning(f"Missing required columns for this chart: {', '.join(missing)}")
            return False
        if sample[list(cols)].dropna().empty:
            st.warning("Not enough non-NaN data to render this chart.")
            return False
        return True

    #Render the chosen chart
    if chart_type == "Period–Radius (scatter, log–log)":
        if need("koi_period", "koi_prad"):
            fig = px.scatter(
                sample, x="koi_period", y="koi_prad",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                hover_data=["koi_model_snr", "koi_kepmag"] if "koi_model_snr" in sample and "koi_kepmag" in sample else None,
                opacity=0.8, title="Period–Radius Diagram"
            )
            fig.update_xaxes(type="log", title="Orbital Period (days)")
            fig.update_yaxes(type="log", title="Planet Radius (Earth radii)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Radius–Insolation (scatter, log X)":
        if need("koi_insol", "koi_prad"):
            fig = px.scatter(
                sample, x="koi_insol", y="koi_prad",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Radius vs Insolation"
            )
            fig.update_xaxes(type="log", title="Insolation (S⊕)")
            fig.update_yaxes(title="Planet Radius (R⊕)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Transit Duration vs Period (scatter)":
        if need("koi_period", "koi_duration"):
            fig = px.scatter(
                sample, x="koi_period", y="koi_duration",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Transit Duration vs Period"
            )
            fig.update_xaxes(title="Orbital Period (days)")
            fig.update_yaxes(title="Transit Duration (hours)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "SNR vs Kepler Magnitude (scatter)":
        if need("koi_model_snr", "koi_kepmag"):
            fig = px.scatter(
                sample, x="koi_kepmag", y="koi_model_snr",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Detection SNR vs Kepler Magnitude"
            )
            fig.update_xaxes(title="Kepler Magnitude (Kp)")
            fig.update_yaxes(title="Model SNR")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Radius vs Equilibrium Temperature (scatter)":
        if need("koi_teq", "koi_prad"):
            fig = px.scatter(
                sample, x="koi_teq", y="koi_prad",
                color=sample["koi_disposition"] if "koi_disposition" in sample.columns else None,
                opacity=0.8, title="Radius vs Equilibrium Temperature"
            )
            fig.update_xaxes(title="Equilibrium Temperature (K)")
            fig.update_yaxes(title="Planet Radius (R⊕)")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Disposition counts (bar)":
        if need("koi_disposition"):
            counts = (sample["koi_disposition"]
                      .astype("string").str.strip()
                      .value_counts(dropna=False)
                      .reset_index())
            counts.columns = ["koi_disposition", "count"]
            fig = px.bar(counts, x="koi_disposition", y="count", title="KOI Disposition Counts")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type.startswith("Histogram:"):
        # map choice to column
        col_map = {
            "Histogram: Period": "koi_period",
            "Histogram: Radius": "koi_prad",
            "Histogram: Transit Depth": "koi_depth",
            "Histogram: Transit Duration": "koi_duration",
            "Histogram: SNR": "koi_model_snr",
            "Histogram: Equilibrium Temperature": "koi_teq",
        }
        col = col_map[chart_type]
        if need(col):
            bins = st.slider("Bins", 5, 120, 40)
            fig = px.histogram(sample, x=col, nbins=bins, title=f"Histogram: {col}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Correlation Heatmap (numeric)":
        # Only numeric cols from our keep set
        num_keep = [c for c in KOI_KEEP_COLS if c in sample.columns and sample[c].dtype != "object"]
        if len(num_keep) < 2:
            st.warning("Need at least two numeric KOI columns for a correlation heatmap.")
        else:
            corr = sample[num_keep].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation (numeric KOI features)")
            st.plotly_chart(fig, use_container_width=True)

    # Expose the prepared dataframe for plotting (if you need elsewhere)
    st.markdown("#### Cleaned KOI dataset")
    st.dataframe(df, use_container_width=True)
'''


