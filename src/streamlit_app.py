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
    # initialize session state for aggregated data
    if 'aggregated_df' not in st.session_state:
        st.session_state['aggregated_df'] = load_data()

    # Sidebar for parameter selection
    st.sidebar.header('Parameters')
    selected_traits = st.sidebar.multiselect(
        'Select Traits',
        options=['LMA', 'T_mesophyll', 'fias_mesophyll', 'T_cw', 'T_cyt', 'T_chloroplast', 'Sm', 'Sc', 'T_leaf', 'D_leaf'],
        default=['T_cw','Sc','T_leaf', 'D_leaf'])

    pft_group_options = list(gm.PFTs.keys())
    selected_pft_group = st.sidebar.selectbox(
        'Select PFT Group',
        options=pft_group_options,
        index=pft_group_options.index('global_set'))

    # Display the selected parameters
    st.write('Selected Traits:', ", ".join(selected_traits))

    pft_list = gm.PFTs[selected_pft_group]
    pft_description = ""
    if len(pft_list) > 1:
        pft_description = f"(which contains these PFTs: {', '.join(gm.PFTs[selected_pft_group])})"
    st.write('Selected PFT group:', selected_pft_group, pft_description)

    # Perform analysis
    if st.button('Perform Analysis'):
        results = gm.CV_with_PFT_and_combination_of_interest(st.session_state['aggregated_df'], gm.PFTs[selected_pft_group], selected_traits, ensemble_size=50, min_rows=50)

        # ~ import pudb; pudb.set_trace()

        predictability_scores_df, importances_df = dict_to_tables(results)
        # show predictability scores table
        st.subheader('Predictability scores')
        st.dataframe(predictability_scores_df)

        # show importances table
        st.subheader('Gini importances')
        st.dataframe(importances_df)

if __name__ == '__main__':
    main()
