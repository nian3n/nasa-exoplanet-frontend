import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import sys

from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Data Exploration", page_icon="📊", layout="wide")

apply_theme(page_key="sage", nebula_path="assets/nebula.jpg")

header("📊 Data Exploration", "Visualize and analyze raw exoplanet datasets")

uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="exploration_csv")

st.markdown('<div class="glass"><h3>Exploration Dashboard</h3></div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def detect_datetime_columns(df: pd.DataFrame, min_ratio: float = 0.7):
    date_cols = []
    for c in df.columns:
        # quick accept by name hint
        cl = str(c).lower()
        if any(h in cl for h in ["date", "time", "epoch", "disc", "jd", "mjd"]):
            try:
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                if s.notna().mean() > 0.2:  # looser threshold if name hints match
                    date_cols.append(c)
                    continue
            except Exception:
                pass
        
        if df[c].dtype == "object":
            try:
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                if s.notna().mean() >= min_ratio:
                    date_cols.append(c)
            except Exception:
                pass
    return date_cols

def detect_columns(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    dates   = detect_datetime_columns(df)
    # categorical: text-like OR low-cardinality numeric
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    for c in numeric:
        nunq = df[c].nunique(dropna=True)
        if nunq <= min(30, max(8, len(df)//50)):
            if c not in categorical:
                categorical.append(c)
    return numeric, categorical, dates

def suggested_numeric(numeric_cols):
    hints = ["period", "prad", "radius", "mass", "teq", "temp", "sma", "insol", "ecc", "snr", "depth"]
    for h in hints:
        for c in numeric_cols:
            if h in c.lower():
                return c
    return numeric_cols[0] if numeric_cols else None

def suggested_xy(numeric_cols):
    xhint, yhint = None, None
    for c in numeric_cols:
        lc = c.lower()
        if xhint is None and any(k in lc for k in ["period", "per", "sma", "a"]):
            xhint = c
        if yhint is None and any(k in lc for k in ["prad", "radius", "mass", "teq", "temp", "depth", "snr"]):
            yhint = c
        if xhint and yhint and xhint != yhint:
            break
 
    if not xhint and len(numeric_cols) >= 1: xhint = numeric_cols[0]
    if not yhint and len(numeric_cols) >= 2: yhint = numeric_cols[1]
    if xhint == yhint and len(numeric_cols) >= 2:
        yhint = numeric_cols[1]
    return xhint, yhint

#after file is uploaded
if uploaded:
    df = load_csv(uploaded)
    st.success(f"Loaded **{df.shape[0]:,} rows** × **{df.shape[1]} columns**")

    
    num_cols, cat_cols, date_cols = detect_columns(df)

    with st.expander("Detected columns", expanded=False):
        st.write("**Numeric:**", num_cols or "—")
        st.write("**Date/Time-like:**", date_cols or "—")
        st.write("**Categorical (incl. low-cardinality numeric):**", cat_cols or "—")

    with st.sidebar:
        st.header("Chart Controls")
        lib = st.radio("Library", ["Plotly (interactive)", "Matplotlib"], index=0)

        chart_type = st.selectbox(
            "Chart type",
            ["Line", "Scatter", "Histogram", "Box", "Bar (category counts)", "Correlation Heatmap"],
            index=0
        )

        row_limit = st.slider("Max rows (for speed)", min_value=1000, max_value=200000, value=min(50000, len(df)), step=1000)
        sample_df = df.head(row_limit)

    
        idx_opt = "(index)"
        if chart_type == "Line":
            y_default = suggested_numeric(num_cols)
            y_col = st.selectbox("Y (numeric)", [y_default] + [c for c in num_cols if c != y_default] if y_default else num_cols)
            x_choices = [idx_opt] + date_cols + cat_cols + num_cols
            x_col = st.selectbox("X (optional)", x_choices, index=0)
            group = st.selectbox("Color / Group (optional)", ["(none)"] + cat_cols)

        elif chart_type == "Scatter":
            x_default, y_default = suggested_xy(num_cols)
            x_col = st.selectbox("X (numeric)", num_cols, index=max(num_cols.index(x_default) if x_default in num_cols else 0, 0)) if num_cols else None
            y_col = st.selectbox("Y (numeric)", num_cols, index=max(num_cols.index(y_default) if y_default in num_cols else (1 if len(num_cols) > 1 else 0), 0)) if num_cols else None
            group = st.selectbox("Color (optional)", ["(none)"] + cat_cols)

        elif chart_type == "Histogram":
            h_col = st.selectbox("Column (numeric)", num_cols)
            bins = st.slider("Bins", 5, 100, 30)

        elif chart_type == "Box":
            val = st.selectbox("Value (numeric)", num_cols)
            by  = st.selectbox("Group by (category)", cat_cols) if cat_cols else None
            show_points = st.checkbox("Show all points (Plotly only)", value=True)

        elif chart_type == "Bar (category counts)":
            cat = st.selectbox("Category", cat_cols) if cat_cols else None
            topn = st.slider("Top N", 5, 50, 10)

        elif chart_type == "Correlation Heatmap":
            pass 

        if chart_type == "Line":
            if not num_cols:
                st.warning("No numeric columns found.")
            else:
                if lib.startswith("Plotly"):
                    if x_col == idx_opt:
                        fig = px.line(sample_df.reset_index(), x="index", y=y_col, title=f"Line: {y_col}")
                    else:
                        fig = px.line(sample_df, x=x_col if x_col != idx_opt else None, y=y_col, color=None if group == "(none)" else group,
                                  title=f"Line: {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = plt.figure()
                    if x_col == idx_opt:
                        plt.plot(sample_df[y_col])
                        plt.xlabel("Index")
                    else:
                        plt.plot(sample_df[x_col], sample_df[y_col])
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title(f"Line: {y_col}")
                        st.pyplot(fig)

        elif chart_type == "Scatter":
            if len(num_cols) < 2 or x_col is None or y_col is None:
                st.warning("Need at least two numeric columns for a scatter plot.")
            else:
                if lib.startswith("Plotly"):
                    fig = px.scatter(sample_df, x=x_col, y=y_col,
                                 color=None if group == "(none)" else group,
                                 opacity=0.8, title=f"Scatter: {x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = plt.figure()
                    plt.scatter(sample_df[x_col], sample_df[y_col], alpha=0.7)
                    plt.xlabel(x_col); plt.ylabel(y_col)
                    plt.title(f"Scatter: {x_col} vs {y_col}")
                    st.pyplot(fig)

        elif chart_type == "Histogram":
            if not num_cols:
                st.warning("No numeric columns found.")
            else:
                if lib.startswith("Plotly"):
                    fig = px.histogram(sample_df, x=h_col, nbins=bins, title=f"Histogram: {h_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = plt.figure()
                    plt.hist(sample_df[h_col].dropna(), bins=bins)
                    plt.xlabel(h_col); plt.ylabel("Count")
                    plt.title(f"Histogram: {h_col}")
                    st.pyplot(fig)

        elif chart_type == "Box":
            if not num_cols:
                st.warning("No numeric columns found.")
            elif by is None and lib.startswith("Plotly"):
            # single series (Plotly)
                fig = px.box(sample_df, y=val, points="all" if show_points else False, title=f"Box: {val}")
                st.plotly_chart(fig, use_container_width=True)
            elif by is None:
            # single series (Matplotlib)
                fig = plt.figure()
                plt.boxplot(sample_df[val].dropna())
                plt.ylabel(val); plt.title(f"Box: {val}")
                st.pyplot(fig)
            else:
                if lib.startswith("Plotly"):
                    fig = px.box(sample_df, x=by, y=val, points="all" if show_points else False,
                             title=f"Box: {val} by {by}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                # Grouped box via pandas helper to preserve simple Matplotlib usage
                    fig = plt.figure()
                    sample_df[[by, val]].dropna().boxplot(by=by, grid=False, rot=45)
                    plt.suptitle("")  
                    plt.title(f"Box: {val} by {by}")
                    plt.xlabel(by); plt.ylabel(val)
                    st.pyplot(fig)

        elif chart_type == "Bar (category counts)":
            if not cat_cols or cat is None:
                st.warning("No categorical columns found.")
            else:
                counts = sample_df[cat].astype("string").value_counts(dropna=False).head(topn).reset_index()
                counts.columns = [cat, "count"]
            if lib.startswith("Plotly"):
                fig = px.bar(counts, x=cat, y="count", title=f"Top {topn} {cat} counts")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = plt.figure()
                plt.bar(counts[cat], counts["count"])
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("count"); plt.title(f"Top {topn} {cat} counts")
                st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":
            if len(num_cols) < 2:
                st.warning("Need multiple numeric columns for correlation heatmap.")
            else:
                corr = sample_df[num_cols].corr(numeric_only=True)
            if lib.startswith("Plotly"):
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation (numeric)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = plt.figure()
                plt.imshow(corr, interpolation="nearest")
                plt.colorbar()
                plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
                plt.yticks(range(len(corr.columns)), corr.columns)
                plt.title("Correlation (numeric)")
                plt.tight_layout()
                st.pyplot(fig)

else:
    st.info("Upload a CSV to begin exploring your data.")
