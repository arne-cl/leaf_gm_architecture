import streamlit as st
import numpy as np
import pandas as pd

import leaf_gm_architecture.functions as gm
from leaf_gm_architecture.streamlit_utils import dict_to_tables, PageNames


def app():
    st.header(PageNames.CROSSVAL.value)

    with st.form(key='predictability_form'):
        selected_traits = st.multiselect(
            'Select Traits',
            options=['LMA', 'T_mesophyll', 'fias_mesophyll', 'T_cw', 'T_cyt', 'T_chloroplast', 'Sm', 'Sc', 'T_leaf', 'D_leaf'],
            default=['T_cw', 'Sc', 'T_leaf', 'D_leaf'])

        ensemble_size = st.number_input('Ensemble Size', min_value=1, value=50, step=1)
        min_rows = st.number_input('Minimum Rows', min_value=1, value=50, step=1)

        pft_group_options = list(gm.PFTs.keys())
        selected_pft_group = st.selectbox(
            'Select PFT Group',
            options=pft_group_options,
            index=pft_group_options.index('global_set'))
        
        submit_button = st.form_submit_button('Perform Analysis')

    if submit_button:
        st.write('Selected Traits:', ", ".join(selected_traits))
        pft_list = gm.PFTs[selected_pft_group]
        pft_description = ""
        if len(pft_list) > 1:
            pft_description = f"(which contains these PFTs: {', '.join(pft_list)})"
        st.write('Selected PFT group:', selected_pft_group, pft_description)

        results, model = gm.CV_with_PFT_and_combination_of_interest(st.session_state['aggregated_df'], gm.PFTs[selected_pft_group], selected_traits, ensemble_size=ensemble_size, min_rows=min_rows)
        predictability_scores_df, importances_df = dict_to_tables(results)

        # show predictability scores table
        st.subheader('Predictability scores')
        st.dataframe(predictability_scores_df)

        # show importances table
        st.subheader('Gini importances')
        st.dataframe(importances_df)

        # User input for prediction
        st.subheader('Predict gm value')
        inputs = {}
        for trait in selected_traits:
            min_val = st.session_state['global_df'][trait].min()
            max_val = st.session_state['global_df'][trait].max()
            inputs[trait] = st.number_input(f'Insert {trait} (range: {min_val} to {max_val})', min_value=float(min_val), max_value=float(max_val))

        if st.button('Predict gm'):
            X = np.array(list(inputs.values())).reshape(1, -1)
            gm_val = model.predict(X)[0]
            st.write(f'Predicted gm value: {np.round(gm_val, 3)}')
