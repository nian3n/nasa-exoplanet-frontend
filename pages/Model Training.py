import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Model Training", page_icon="🤖", layout="wide")

apply_theme(page_key="ochre", nebula_path="assets/nebula.jpg")

header("🤖 Model Training", "Upload training data and build models")

st.file_uploader("Upload training data (CSV)", type=["csv"], key="training_csv")

# st.markdown('<div class="glass"><h3>Training Interface</h3></div>', unsafe_allow_html=True)  # commented out, replaced by image glass div below

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

img_path = "nasa-exoplanet-frontend-main/assets/model_training_dashboard.jpg"
data_uri = get_data_uri_img(img_path)
if data_uri:
	st.markdown(
		f"""
		<div class='glass' style="background-image: url('{data_uri}'); background-size: cover; background-position: center; min-height: 200px; border-radius:1rem; overflow: hidden; margin-bottom:1.5rem;">
			<div style='backdrop-filter: blur(6px); padding:1.5rem; color:#fff;'>
				<h3 style='margin-top:0;'>Training Interface</h3>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
# =====================
# END IMAGE GLASS DIV BLOCK
# =====================

