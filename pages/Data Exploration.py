

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from telescopes.main_info import info
from utils.general_description import description
from theme import apply_theme, header

st.set_page_config(page_title="ExoLab • Data Exploration", page_icon="📊", layout="wide")

apply_theme(page_key="sage", nebula_path="assets/nebula.jpg")

header("📊 Data Exploration", "Visualize and analyze raw exoplanet datasets")

st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="exploration_csv")

st.markdown('<div class="glass"><h3>Exploration Dashboard</h3></div>', unsafe_allow_html=True)


types = st.sidebar.multiselect(
        "Select type of chart to display",
        list(info.keys()),
        default=list(info.keys())
    )

for t in types:
    # Display logo
    st.image(f'./data/logos/{t}.jpeg')
    st.markdown(description[t]['general_description'])

    # Euclid is for now the only mission that I have completed. Message elsewhere
    if t != 'Euclid':
        complete="(Non exhaustive list)"
    else:
        complete=""

    # Loop through the instruments
    # add warning if not complete
    st.markdown('### Instruments '+complete+':')
    for instrument in list(description[telescope]['Instruments'].keys()):
        st.markdown("- " + description[telescope]['Instruments'][instrument])
    
    # Loop through the instruments
    # add warning if not complete
    st.markdown('### Surveys '+complete+':')
    for survey in list(description[telescope]['Surveys'].keys()):
        st.markdown("- " + description[telescope]['Surveys'][survey])