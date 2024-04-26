import streamlit as st
import pandas as pd
import Leaf_gm_architecture_functions as gm


@st.cache_data  # this caches the aggregated input data between runs
def load_data():
    return pd.read_parquet('gm_dataset_Knauer_et_al_2022_aggregated.parquet')


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
    
    if isinstance(results, pd.DataFrame):
        st.write(results)
        results.to_csv('results.csv', index=False)
        st.success('Analysis complete and results saved!')
    elif isinstance(results, dict):
        st.json(results)  # Display the results as JSON in the app
        st.error('Analysis results are not in tabular form. See the JSON output.')
    else:
        st.error('Unexpected result type: ' + str(type(results)))
