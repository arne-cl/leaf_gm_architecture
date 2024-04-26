import streamlit as st
import pandas as pd
import Leaf_gm_architecture_functions as gm

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_parquet('gm_dataset_Knauer_et_al_2022_aggregated.parquet')

# ~ aggregated_df = load_data()

# Initialize session state for aggregated data
if 'aggregated_df' not in st.session_state:
    st.session_state['aggregated_df'] = load_data()

# ~ combination_of_traits = ['T_cw','Sc','T_leaf', 'D_leaf']

# Sidebar for parameter selection
st.sidebar.header('Parameters')
selected_traits = st.sidebar.multiselect(
    'Select Traits',
    options=['LMA', 'T_mesophyll', 'fias_mesophyll', 'T_cw', 'T_cyt', 'T_chloroplast', 'Sm', 'Sc', 'T_leaf', 'D_leaf'],
    default=['T_cw','Sc','T_leaf', 'D_leaf'])

pft_group_options = list(gm.PFTs.keys())
selected_pft = st.sidebar.selectbox(
    'Select PFT Group',
    options=pft_group_options,
    index=pft_group_options.index('global_set'))

# Button to perform aggregation
# ~ if st.button('Aggregate Data'):
    # ~ st.session_state['aggregated_df'] = gm.data_aggregation(df)
    # ~ st.write('Data Aggregated!')

# Display the selected parameters
st.write('Selected Traits:', selected_traits)
st.write('Selected PFT:', selected_pft)

# Perform analysis
if st.button('Perform Analysis'):
    if st.session_state['aggregated_df'] is not None:
        results = gm.CV_with_PFT_and_combination_of_interest(st.session_state['aggregated_df'], gm.PFTs[selected_pft], selected_traits, ensemble_size=50, min_rows=50)
        
        if isinstance(results, pd.DataFrame):
            st.write(results)
            results.to_csv('results.csv', index=False)
            st.success('Analysis complete and results saved!')
        elif isinstance(results, dict):
            st.json(results)  # Display the results as JSON in the app
            st.error('Analysis results are not in tabular form. See the JSON output.')
        else:
            st.error('Unexpected result type: ' + str(type(results)))
    else:
        st.error('Please aggregate the data first.')

# To run this app, save this script as app.py and in your terminal run: streamlit run app.py
