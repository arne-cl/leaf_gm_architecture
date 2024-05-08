import streamlit as st
import numpy as np
import pandas as pd

import Leaf_gm_architecture_functions as gm


@st.cache_data  # this caches the aggregated input data between runs
def load_data():
    return pd.read_parquet('gm_dataset_Knauer_et_al_2022_aggregated.parquet')


def dict_to_tables(results):
    """
    Helper function to convert CV_with_PFT_and_combination_of_interest()
    result dictionary into individual dataframes.

    Parameters
    ----------
    results : dict
        The resulting predictability scores and importance of the given
        traits calculated in CV_with_PFT_and_combination_of_interest().
        
    Returns 
    -------
    predictability_scores_df : pd.DataFrame
        DataFrame with one column per predictability score
    importances_df : pd.DataFrame
        DataFrame with one column per importance value
    """
    # create predictability scores DataFrame by excluding 'importances'
    predictability_scores_df = pd.DataFrame({k: [v] for k, v in results.items() if k != 'importances'})

    if results['importances'] is np.nan:
        importances_df = pd.DataFrame({'Value': results['importances']}, index=[0])
    else:
        importances_df = pd.DataFrame({'Value': results['importances']})

    return predictability_scores_df, importances_df


def main():
    st.title('leaf_gm_architecture')

    # initialize session state for aggregated data
    if 'aggregated_df' not in st.session_state:
        st.session_state['aggregated_df'] = load_data()

    # create tabs for each example use case
    tab_name_cross_val = "Cross Validation"
    tab_name_cross_pred_global_pft = "Cross Prediction: global-PFT"
    tab_name_cross_pred_pft_pft = "Cross Prediction: PFT-PFT"
    tab_cross_val, tab_cross_pred_global_pft, tab_cross_pred_pft_pft = st.tabs([tab_name_cross_val, tab_name_cross_pred_global_pft, tab_name_cross_pred_pft_pft])

    with tab_cross_val:
        st.header(tab_name_cross_val)
        
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

            results = gm.CV_with_PFT_and_combination_of_interest(st.session_state['aggregated_df'], gm.PFTs[selected_pft_group], selected_traits, ensemble_size=ensemble_size, min_rows=min_rows)
            predictability_scores_df, importances_df = dict_to_tables(results)

            # show predictability scores table
            st.subheader('Predictability scores')
            st.dataframe(predictability_scores_df)

            # show importances table
            st.subheader('Gini importances')
            st.dataframe(importances_df)

    with tab_cross_pred_global_pft:
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
            else:
                st.write("There are no trained models because of data availability!")

        if st.button('Calculate Cross Prediction: global-PFT with combination of interest'):
            # TODO: call cross_prediction_global_PFT_with_combination_of_interest
            global_PFT_comb_interest_results = gm.cross_prediction_global_PFT_with_combination_of_interest(
                df_agg=st.session_state['aggregated_df'],
                PFT_of_interest=['ferns'],
                combination_of_interest=selected_traits, 
                ensemble_size=ensemble_size,
                minimum_train_rows=min_train_rows, 
                minimum_test_rows=min_test_rows)

            st.write(global_PFT_comb_interest_results)

            # ~ if global_PFT_comb_interest_results.shape[0] > 0:
                # ~ st.write("Trained models:", global_PFT_comb_interest_results.shape[0])
                # ~ st.write(table_of_results)
                # ~ average_imps_g, average_values_c = gm.total_importances(global_PFT_comb_interest_results)
                # ~ st.write("The IMP_G of the contributing traits in the trained models:", average_imps_g)
                # ~ st.write("The IMP_C of the contributing traits in the trained models:", average_values_c)
            # ~ else:
                # ~ st.write("There are no trained models because of data availability!")

    with tab_cross_pred_pft_pft:
        st.header(tab_name_cross_pred_pft_pft)

        # TODO: call this function as well
        # cross_prediction_global_PFT_with_combination_of_interest
        st.write("FIXME: This tab wasn't implemented, yet!")

    # ~ cross_prediction_global_PFT_with_combination_of_interest(
        # ~ df_agg,PFT_of_interest,
        # ~ combination_of_interest, # different
        # ~ # these params are the same, but the ensemble_size value is different
        # ~ ensemble_size=5,  minimum_train_rows=40,minimum_test_rows=10)
        table_of_results = gm.cross_prediction_PFT_PFT_with_combination_of_interest(
            df_agg=st.session_state['aggregated_df'],
            PFT_train=['ferns'], # TODO: fix these values
            PFT_test=['ferns'], # TODO: fix these values
            combination_of_interest=selected_traits, 
            ensemble_size=ensemble_size,
            minimum_train_rows=min_train_rows, 
            minimum_test_rows=min_test_rows)

if __name__ == '__main__':
    main()

# TODO: implement these, but fix the ordering/matching first
# second tab should only contain global-PFT, third tab only PFT-PFT
# cross_prediction_PFT_PFT_with_combination_of_interest # add to tab2; name: cross prediction PFT-PFT
# cross_prediction_global_PFT_with_combination_of_interest # new tab3; name: cross prediction global-PFT
