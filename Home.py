import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Home", page_icon="🪐", layout="wide")

# Pick a themed color + nebula header image
apply_theme(page_key="deep_teal", nebula_path="assets/nebula.jpg")

# Nebula hero with title + subtitle
header("🪐 ExoHorizon ", "Best ML tool to explore new worlds beyond our own")

# Content
page_descriptions = {
    "About": "Learn about who we are and how ExoHorizon is making impacts.",
    "Data Exploration": "Dive into real exoplanet catalogs with interactive charts and visualizations.",
    "Model Evaluation": "See how our ML model performs.",
    "Model Training": "Upload your dataset to train our pre-configured exoplanet model and track results."
}

st.markdown("### Explore ExoHorizon")


cols = st.columns(2, gap="large")

for idx, (page, desc) in enumerate(page_descriptions.items()):
    target = f"pages/{page}.py"                     
    with cols[idx % 2]:
        st.markdown(
            f"""
            <div class="glass" style="padding:1.5rem; margin-bottom:1.5rem; border-radius:1rem;">
              <h3>{page}</h3>
              <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Native Streamlit link (works with multipage routing)
        st.page_link(target, label=f"Go to {page}", icon="➡️")