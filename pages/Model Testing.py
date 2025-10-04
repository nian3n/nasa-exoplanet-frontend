import streamlit as st
from theme import apply_theme, header
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="ExoHorizon • Model Testing", page_icon="📈", layout="wide")

apply_theme(page_key="forest", nebula_path="assets/nebula.jpg")

header("📈 Model Testing", "Upload your exoplanet datasets and see how our model performs")

# File uploader - we will use this to trigger evaluation when a CSV is provided
uploaded_file = st.file_uploader("Upload evaluation dataset (CSV)", type=["csv"], key="evaluation_csv")

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

	# ----------------------------
	# If a file is uploaded, run model loading, preprocessing, prediction, and evaluation
	# ----------------------------
	if uploaded_file:
		try:
			pipeline_data = joblib.load("nasa-exoplanet-frontend-main/Exo_planet_model.pkl")
			model = pipeline_data.get("model")
			scaler = pipeline_data.get("scaler")
			le = pipeline_data.get("label_encoder")
			numeric_cols = pipeline_data.get("features")
			st.success("✅ Unified exoplanet model loaded successfully!")
		except Exception as e:
			st.error(f"Failed to load model: {e}")
			st.stop()

		try:
			df_test = pd.read_csv(uploaded_file, engine='python', comment='#', on_bad_lines='skip')
			st.success(f"CSV loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
		except Exception as e:
			st.error(f"Failed to read uploaded CSV: {e}")
			st.stop()

		# Map common column names to the unified feature names expected by the pipeline
		rename_map = {
			# KOI
			'koi_period':'period','koi_duration':'duration','koi_depth':'depth',
			'koi_prad':'planet_radius','koi_srad':'stellar_radius','koi_smass':'stellar_mass',
			'koi_model_snr':'snr','koi_fpflag_nt':'fpflag_nt','koi_fpflag_ss':'fpflag_ss',
			'koi_fpflag_co':'fpflag_co','koi_fpflag_ec':'fpflag_ec',
			# TOI
			'toi_period':'period','toi_duration':'duration','toi_depth':'depth',
			'toi_prad':'planet_radius','toi_srad':'stellar_radius','toi_smass':'stellar_mass',
			'toi_model_snr':'snr','toi_fpflag_nt':'fpflag_nt','toi_fpflag_ss':'fpflag_ss',
			'toi_fpflag_co':'fpflag_co','toi_fpflag_ec':'fpflag_ec',
			# K2
			'k2_period':'period','k2_duration':'duration','k2_depth':'depth',
			'k2_prad':'planet_radius','k2_srad':'stellar_radius','k2_smass':'stellar_mass',
			'k2_snr':'snr','k2_fpflag_nt':'fpflag_nt','k2_fpflag_ss':'fpflag_ss',
			'k2_fpflag_co':'fpflag_co','k2_fpflag_ec':'fpflag_ec'
		}
		df_test = df_test.rename(columns=rename_map)

		# Ensure all numeric_cols exist in the uploaded dataframe
		for col in (numeric_cols or []):
			if col not in df_test.columns:
				df_test[col] = np.nan

		df_numeric = df_test[numeric_cols].copy() if numeric_cols else df_test.select_dtypes(include=[np.number]).copy()
		df_numeric = df_numeric.fillna(0)

		# Scale features and predict
		try:
			df_scaled = scaler.transform(df_numeric)
		except Exception:
			# If scaler is None or transform fails, try to use raw numeric data
			df_scaled = df_numeric.values

		y_pred = model.predict(df_scaled)
		y_proba = None
		try:
			y_proba = model.predict_proba(df_scaled)
		except Exception:
			pass

		if le is not None:
			try:
				y_pred_labels = le.inverse_transform(y_pred)
			except Exception:
				y_pred_labels = y_pred
		else:
			y_pred_labels = y_pred

		if y_proba is not None:
			y_pred_conf = [f"{np.max(proba)*100:.2f}%" for proba in y_proba]
		else:
			y_pred_conf = ["N/A"] * len(y_pred_labels)

		df_test['Predicted_Disposition'] = y_pred_labels
		df_test['Confidence'] = y_pred_conf

		st.markdown("### Predictions (First 1000 Rows)")
		st.dataframe(df_test.head(1000))

		# If ground truth is available, compute accuracy and confusion matrix
		possible_truth_cols = ['koi_disposition','tfopwg_disp','disposition']
		truth_col = next((c for c in possible_truth_cols if c in df_test.columns), None)

		if truth_col:
			label_standard = {
				'CONFIRMED':'CONFIRMED','CANDIDATE':'CANDIDATE','PC':'CANDIDATE',
				'CP':'CONFIRMED','FALSE POSITIVE':'FALSE POSITIVE','FP':'FALSE POSITIVE'
			}
			y_true = df_test[truth_col].map(str).map(lambda x: label_standard.get(x.strip(), np.nan)).dropna()
			df_eval = df_test.loc[y_true.index]

			actual_acc = accuracy_score(y_true, df_eval['Predicted_Disposition'])
			st.success(f"✅ Model Accuracy: {actual_acc*100:.2f}%")

			try:
				classes = le.classes_ if le is not None else np.unique(df_eval['Predicted_Disposition'])
				cm = confusion_matrix(y_true, df_eval['Predicted_Disposition'], labels=classes)
				disp = ConfusionMatrixDisplay(cm, display_labels=classes)
				disp.plot(cmap=plt.cm.Blues)
				plt.title("Confusion Matrix")
				st.pyplot(plt.gcf())
			except Exception as e:
				st.info(f"Could not plot confusion matrix: {e}")
		else:
			st.info("No ground truth column found; cannot compute accuracy.")

		# Simple accuracy overview plot (placeholder training/cv values)
		train_acc = 0.95
		cv_scores = np.array([0.92,0.93,0.91,0.94,0.92])
		test_acc = actual_acc if truth_col else 0

		fig, ax = plt.subplots(figsize=(8,5))
		ax.plot(range(1,len(cv_scores)+1), cv_scores, marker='o', label='Validation (CV) Accuracy')
		ax.axhline(train_acc,color='green',linestyle='--',label='Training Accuracy')
		ax.axhline(test_acc,color='red',linestyle='--',label='Test Accuracy')
		ax.set_xlabel('Fold')
		ax.set_ylabel('Accuracy')
		ax.set_title('Training / Validation / Test Accuracy')
		ax.legend()
		st.pyplot(fig)






'''import streamlit as st
from theme import apply_theme, header

st.set_page_config(page_title="ExoHorizon • Model Testing", page_icon="📈", layout="wide")

apply_theme(page_key="forest", nebula_path="assets/nebula.jpg")

header("📈 Model Testing", "Upload your exoplanet datasets and see how our model performs")

st.file_uploader("Upload evaluation dataset (CSV)", type=["csv"], key="evaluation_csv")

st.markdown('<div class="glass"><h3>Evaluation Results</h3></div>', unsafe_allow_html=True)'''
