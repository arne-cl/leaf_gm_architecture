import streamlit as st
import numpy as np
import pandas as pd

import Leaf_gm_architecture_functions as gm
from Leaf_gm_architecture_functions.streamlit_app import load_data

def main():
    st.title('leaf_gm_architecture')

    # initialize session state for aggregated data
    if 'aggregated_df' not in st.session_state:
        st.session_state['aggregated_df'] = load_data()

    tab_name_cross_pred_global_pft = "Cross Prediction: global-PFT"

    st.header(tab_name_cross_pred_global_pft)
    selected_traits = st.multiselect('Select traits', 
                                     ['LMA', 'T_mesophyll', 'fias_mesophyll', 'T_cw', 'T_cyt', 'T_chloroplast', 'Sm', 'Sc', 'T_leaf', 'D_leaf'],
                                     default=['LMA', 'T_mesophyll', 'fias_mesophyll', 'T_cw', 'T_cyt', 'T_chloroplast', 'Sm', 'Sc', 'T_leaf', 'D_leaf'])
    ensemble_size = st.number_input('Ensemble Size', min_value=1, value=5, step=1)
    min_train_rows = st.number_input('Minimum Train Rows', min_value=1, value=40, step=1)
    min_test_rows = st.number_input('Minimum Test Rows', min_value=1, value=10, step=1)

    if st.button('Calculate Cross Prediction: global-PFT'):
        global_PFT_results = gm.cross_prediction_global_PFT(
            df_agg=st.session_state['aggregated_df'],
            PFT_of_interest=['ferns'],
            traits_list=selected_traits, 
            ensemble_size=ensemble_size,
            minimum_train_rows=min_train_rows, 
            minimum_test_rows=min_test_rows)

        if global_PFT_results.shape[0] > 0:
            st.write("Trained models:", global_PFT_results.shape[0])
            st.write(global_PFT_results)
            average_imps_g, average_values_c = gm.total_importances(global_PFT_results)
            st.write("The IMP_G of the contributing traits in the trained models:", average_imps_g)
            st.write("The IMP_C of the contributing traits in the trained models:", average_values_c)

            # extract unique traits from global_PFT_results
            trait_options = set()
            for traits_list in global_PFT_results['Traits']:
                trait_options.update(traits_list)
            trait_options = sorted(trait_options)

            # let user select combination of traits from the extracted options
            selected_traits = st.multiselect('Select combination of traits for detailed analysis',
                                             options=trait_options)

            if selected_traits:
                global_PFT_comb_interest_results = gm.cross_prediction_global_PFT_with_combination_of_interest(
                    df_agg=st.session_state['aggregated_df'],
                    PFT_of_interest=['ferns'],
                    combination_of_interest=selected_traits, 
                    ensemble_size=ensemble_size,
                    minimum_train_rows=min_train_rows, 
                    minimum_test_rows=min_test_rows)

                st.header(tab_name_cross_pred_global_pft + " (with combination of interest)")
                st.write(global_PFT_comb_interest_results)
            else:
                st.error("Please select at least one trait to perform a detailed analysis.")
        else:
            st.write("There are no trained models because of data availability!")

if __name__ == '__main__':
    main()
