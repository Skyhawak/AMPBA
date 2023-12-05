import streamlit as st
import pandas as pd
import numpy as np
from autots import AutoTS
import matplotlib.pyplot as plt

def preprocess_data_for_customer_weekly(data, customer_name, start_date, end_date='2023-08-23'):
    customer_data = data[data['Customer Name (Cleaned)'] == customer_name]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    customer_data = customer_data[(customer_data['Date'] >= start_date) & (customer_data['Date'] <= end_date)]
    customer_data = customer_data.groupby('Date').agg({'QTY': 'sum'}).reset_index()
    return customer_data

def aggregate_weekly(preprocessed_data_weekly):
    preprocessed_data_weekly = preprocessed_data_weekly.set_index('Date')
    weekly_data = preprocessed_data_weekly.resample('W').sum().reset_index()
    return weekly_data

def impute_weekly_data(weekly_data, imputation_method):
    weekly_data['QTY'] = weekly_data['QTY'].replace(0, np.nan)
    if imputation_method == 'ffill':
        weekly_data['QTY'] = weekly_data['QTY'].fillna(method='ffill')
    elif imputation_method == 'bfill':
        weekly_data['QTY'] = weekly_data['QTY'].fillna(method='bfill')
    else:
        weekly_data['QTY'] = weekly_data['QTY'].interpolate(method=imputation_method)
    return weekly_data

# Streamlit UI setup
st.title("Customer Sales Forecasting")

uploaded_file = st.file_uploader("Upload your sales data file (Excel format)", type=['xlsx'])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Customer Name (Cleaned)', 'Date', 'QTY']]

    customer_list = data['Customer Name (Cleaned)'].unique()
    selected_customer = st.selectbox('Select a Customer', customer_list)

    imputation_methods = ['ffill', 'bfill', 'akima', 'linear', 'cubic']
    imputation_method = st.selectbox('Select an Imputation Method', imputation_methods)

    confidence_interval = st.slider("Select Confidence Interval", 0.80, 0.99, 0.95)

    if st.button('Forecast Sales'):
        start_date = '2021-11-01'
        preprocessed_data = preprocess_data_for_customer_weekly(data, selected_customer, start_date)
        weekly_data = aggregate_weekly(preprocessed_data)
        imputed_data = impute_weekly_data(weekly_data, imputation_method)

        available_data_length = len(imputed_data)
        forecast_length = int(len(imputed_data) * 0.2)  # Use 80% of data for training

        model = AutoTS(forecast_length=forecast_length, frequency='W', prediction_interval=confidence_interval)
        model = model.fit(imputed_data, date_col='Date', value_col='QTY')

        prediction = model.predict(forecast_length=forecast_length)
        forecast_df = prediction.forecast

        # Extracting upper and lower confidence intervals
        upper_confidence_interval = prediction.upper_forecast
        lower_confidence_interval = prediction.lower_forecast

        # Combine forecast with confidence intervals
        forecast_combined = forecast_df.join(upper_confidence_interval, rsuffix='_upper').join(lower_confidence_interval, rsuffix='_lower')

        # Display forecast data with confidence intervals
        st.write("Forecasted Data with Confidence Intervals:")
        st.dataframe(forecast_combined.head(forecast_length))

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(imputed_data['Date'], imputed_data['QTY'], label='Original Data')
        ax.plot(forecast_df.index, forecast_df['QTY'], label='Forecasted Data', linestyle='--')
        ax.fill_between(forecast_df.index, lower_confidence_interval['QTY'], upper_confidence_interval['QTY'], color='gray', alpha=0.3, label='Confidence Interval')
        ax.legend()
        ax.set_title('Original vs Forecasted Data (Weekly)')
        ax.set_xlabel('Date')
        ax.set_ylabel('QTY')
        st.pyplot(fig)

        # Display model details and error metrics
        model_results = model.results()
        best_model_results = model_results.iloc[0]  # Assuming the best model is at the top
        st.write("Best Model Details:")
        st.table(best_model_results[['Model', 'ModelParameters']])
        st.write("Error Metrics for the Best Model:")
        error_metrics_df = best_model_results.drop(['Model', 'ID', 'Generation', 'ModelParameters'])
        st.table(error_metrics_df)

# To run this script, save it as a Python file and execute 'streamlit run your_script.py'.
