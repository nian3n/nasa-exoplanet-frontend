import streamlit as st
import base64
import os
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Home", page_icon="🪐", layout="wide")

# Pick a themed color + nebula header image
apply_theme(page_key="deep_teal", nebula_path="assets/nebula.jpg")

# Nebula hero with title + subtitle
header("🪐 ExoHorizon ", "Best ML tool to explore new worlds beyond our own")

# ---------- helper: load & cache data-uri ----------
@st.cache_data(show_spinner=False)
def get_data_uri(path: str, fallback: str = "assets/nebula.jpg") -> str:
    """Read image file and return a data URI. Falls back if file missing."""
    if not os.path.exists(path):
        path = fallback
        if not os.path.exists(path):
            # Use about.jpg as a last-resort fallback
            path = "nasa-exoplanet-frontend-main/assets/about.jpg"
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


# Page descriptions and image mapping
page_descriptions = {
    "About": "Learn about who we are and how ExoHorizon is making impacts.",
    "Data Exploration": "Dive into real exoplanet catalogs with interactive charts and visualizations.",
    "Model Testing": "See how our ML model performs.",
    "Model Training": "Upload your dataset to train our pre-configured exoplanet model and track results."
}

# Map pages to image files
page_images = {
    "About": "nasa-exoplanet-frontend-main/assets/about.jpg",
    "Data Exploration": "nasa-exoplanet-frontend-main/assets/data_exploration.jpg",
    "Model Testing": "nasa-exoplanet-frontend-main/assets/model_testing.jpg",
    "Model Training": "nasa-exoplanet-frontend-main/assets/model_training.jpg"
}

st.markdown("### Explore ExoHorizon")

# ===========================================================
# IMAGE-BACKED CARDS (replaces original divs)
# ===========================================================
cols = st.columns(2, gap="large")

for idx, (page, desc) in enumerate(page_descriptions.items()):
    target = f"pages/{page}.py"
    img_path = page_images.get(page, "assets/nebula.jpg")
    data_uri = get_data_uri(img_path)

    with cols[idx % 2]:
        html_card = (
            f"<div class='glass' style=\"background-image: url('{data_uri}'); background-size: cover; background-position: center; min-height: 200px; border-radius:1rem; overflow: hidden; margin-bottom:1.5rem;\">"
            f"<div style='backdrop-filter: blur(6px); padding:1.5rem; color:#fff;'>"
            f"<h3 style='margin-top:0;'>{page}</h3>"
            f"<p style='margin:0;'>{desc}</p>"
            f"</div></div>"
        )
        st.markdown(html_card, unsafe_allow_html=True)
        # Native Streamlit link (works with multipage routing)
        st.page_link(target, label=f"Go to {page}", icon="➡️")
# ===========================================================
# END OF IMAGE-BACKED CARDS
# ===========================================================

# If you want to revert back to plain divs, just comment out the entire block above
# and uncomment your original loop from before.






