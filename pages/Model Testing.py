import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Model Testing", page_icon="📈", layout="wide")

apply_theme(page_key="forest", nebula_path="assets/nebula.jpg")

header("📈 Model Testing", "Upload your exoplanet datasets and see how our model performs")

st.file_uploader("Upload evaluation dataset (CSV)", type=["csv"], key="evaluation_csv")

# st.markdown('<div class="glass"><h3>Evaluation Results</h3></div>', unsafe_allow_html=True)  # commented out, replaced by image glass div below

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

img_path = "nasa-exoplanet-frontend-main/assets/model_testing_dashboard.jpg"
data_uri = get_data_uri_img(img_path)
if data_uri:
	st.markdown(
		f"""
		<div class='glass' style="background-image: url('{data_uri}'); background-size: cover; background-position: center; min-height: 200px; border-radius:1rem; overflow: hidden; margin-bottom:1.5rem;">
			<div style='backdrop-filter: blur(6px); padding:1.5rem; color:#fff;'>
				<h3 style='margin-top:0;'>Evaluation Results</h3>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
# =====================
# END IMAGE GLASS DIV BLOCK
# =====================



'''import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Model Testing", page_icon="📈", layout="wide")

apply_theme(page_key="forest", nebula_path="assets/nebula.jpg")

header("📈 Model Testing", "Upload your exoplanet datasets and see how our model performs")

st.file_uploader("Upload evaluation dataset (CSV)", type=["csv"], key="evaluation_csv")

st.markdown('<div class="glass"><h3>Evaluation Results</h3></div>', unsafe_allow_html=True)'''
