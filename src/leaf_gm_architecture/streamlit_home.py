import streamlit as st
from streamlit_option_menu import option_menu

import leaf_gm_architecture
from leaf_gm_architecture.streamlit_utils import load_dataset, load_aggregated_data, PageNames
from leaf_gm_architecture import (
    streamlit_crossval,
    streamlit_crosspred_global_pft,
    streamlit_crosspred_pft_pft)




selected = option_menu(
    menu_title=None,  # required
    options=[page.value for page in PageNames],  # required
    menu_icon="cast",  # optional
    default_index=1,  # optional
    orientation="horizontal",
)


# initialize session state for raw dataset
if 'global_df' not in st.session_state:
    st.session_state['global_df'] = load_dataset()

# initialize session state for aggregated data
if 'aggregated_df' not in st.session_state:
    st.session_state['aggregated_df'] = load_aggregated_data()

if selected == PageNames.ABOUT.value:
    st.title('leaf_gm_architecture')
    st.subheader(f"This is a demo for Milad's paper.")

if selected == PageNames.CROSSVAL.value:
    streamlit_crossval.app()

if selected == PageNames.CROSSPRED_GLOBAL_PFT.value:
    streamlit_crosspred_global_pft.app()

if selected == PageNames.CROSSPRED_PFT_PFT.value:
    streamlit_crosspred_pft_pft.app()
