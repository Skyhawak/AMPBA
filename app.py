import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Function to preprocess data for a specific customer
def preprocess_data_for_customer(data, customer_name):
    customer_data = data[data['Customer Name (Cleaned)'] == customer_name]
    customer_data = customer_data.groupby('Date').agg({'QTY': 'sum'}).reset_index()
    return customer_data

# Function to aggregate data
def aggregate_data(preprocessed_data, frequency):
    if frequency == 'Weekly':
        aggregated_data = preprocessed_data.set_index('Date').resample('W').sum().reset_index()
    elif frequency == 'Biweekly':
        aggregated_data = preprocessed_data.set_index('Date').resample('W').sum().reset_index().iloc[::2, :]
    else:  # Monthly
        aggregated_data = preprocessed_data.set_index('Date').resample('M').sum().reset_index()
    return aggregated_data

# Impute missing values
def impute_data(aggregated_data, imputation_method):
    aggregated_data_imputed = aggregated_data.copy()
    aggregated_data_imputed['QTY'] = aggregated_data_imputed['QTY'].replace(0, np.nan)

    if imputation_method == 'ffill':
        aggregated_data_imputed['QTY'] = aggregated_data_imputed['QTY'].fillna(method='ffill')
    elif imputation_method == 'bfill':
        aggregated_data_imputed['QTY'] = aggregated_data_imputed['QTY'].fillna(method='bfill')
    elif imputation_method in ['akima', 'linear', 'cubic']:
        aggregated_data_imputed['QTY'] = aggregated_data_imputed['QTY'].interpolate(method=imputation_method)

    return aggregated_data_imputed

# Load the dataset (change the file path to your dataset's location)
file_path = 'path_to_your_dataset.xlsx'
data = pd.read_excel(file_path, parse_dates=['Date'])

# Keeping only the required columns
data = data[['Customer Name (Cleaned)', 'Date', 'QTY']]

# Sidebar - Customer selection
customer_name = st.sidebar.selectbox('Select Customer', data['Customer Name (Cleaned)'].unique())

# Sidebar - Aggregation frequency
frequency = st.sidebar.selectbox('Select Aggregation Frequency', ['Weekly', 'Biweekly', 'Monthly'])

# Sidebar - Imputation method
imputation_method = st.sidebar.selectbox('Select Imputation Method', ['None', 'ffill', 'bfill', 'akima', 'linear', 'cubic'])

# Main - Data processing and plotting
if st.sidebar.button('Show Plot'):
    preprocessed_data = preprocess_data_for_customer(data, customer_name)
    aggregated_data = aggregate_data(preprocessed_data, frequency)

    if imputation_method != 'None':
        aggregated_data_imputed = impute_data(aggregated_data, imputation_method)
    else:
        aggregated_data_imputed = aggregated_data

    # Creating the plot
    fig = px.line(aggregated_data, x='Date', y='QTY', title=f'Pallets Ordered - {customer_name}')
    fig.add_scatter(x=aggregated_data_imputed['Date'], y=aggregated_data_imputed['QTY'], mode='lines', name='Imputed Data', line=dict(dash='dash'))
    st.plotly_chart(fig)
