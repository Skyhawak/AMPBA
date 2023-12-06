import streamlit as st
import pandas as pd
from autots import AutoTS
import matplotlib.pyplot as plt
import numpy as np

# Function to load and apply the CSS file
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the CSS
local_css("style.css")

# Function to preprocess data for a specific customer
def preprocess_data_for_customer(data, customer_name, start_date, end_date, frequency):
    customer_data = data[data['Customer Name (Cleaned)'] == customer_name]
    customer_data = customer_data[(customer_data['Date'] >= start_date) & (customer_data['Date'] <= end_date)]
    customer_data = customer_data.groupby('Date').agg({'QTY': 'sum'}).reset_index()
    aggregated_data = customer_data.set_index('Date').resample(frequency).sum().reset_index()
    return aggregated_data

# Function to impute missing values
def impute_data(aggregated_data, imputation_method):
    aggregated_data['QTY'] = aggregated_data['QTY'].replace(0, np.nan)
    if imputation_method == 'ffill':
        aggregated_data['QTY'] = aggregated_data['QTY'].fillna(method='ffill')
    elif imputation_method == 'bfill':
        aggregated_data['QTY'] = aggregated_data['QTY'].fillna(method='bfill')
    else:
        aggregated_data['QTY'] = aggregated_data['QTY'].interpolate(method=imputation_method)
    return aggregated_data

# Function to plot data
def plot_data(original_data, forecast_data, upper_confidence, lower_confidence):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data['Date'], original_data['QTY'], label='Original Data', color='blue')
    plt.plot(forecast_data['Forecast Interval'], forecast_data['Forecast Value'], label='Forecasted Data', color='green', linestyle='--')
    plt.fill_between(forecast_data['Forecast Interval'], lower_confidence, upper_confidence, color='gray', alpha=0.3, label='Confidence Interval')
    plt.title('Original vs Forecasted Data with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('QTY')
    plt.legend()
    plt.tight_layout()  # Adjust layout for better appearance
    return plt

# Streamlit UI
st.title("Wooden Pallets Demand Forecasting")

st.sidebar.header("Input Options")

# File upload
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "csv"])
if uploaded_file is not None:
    # Read file
    if uploaded_file.type == "text/csv":
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    data['Date'] = pd.to_datetime(data['Date'])

    # Select Customer
    customer_name = st.sidebar.selectbox("Select Customer", data['Customer Name (Cleaned)'].unique())

    # Select Aggregation Frequency
    frequency = st.sidebar.selectbox("Select Aggregation Frequency", ['15D', 'Week', 'Month'])

    # Select Imputation Method
    imputation_method = st.sidebar.selectbox("Select Imputation Method", ['ffill', 'bfill', 'linear'])

    # Select Confidence Interval
    confidence_interval = st.sidebar.slider("Select Confidence Interval", 0.0, 1.0, 0.9)

    if st.sidebar.button("Forecast"):
        with st.spinner('Running the model...'):
            start_date = data['Date'].min()
            end_date = data['Date'].max()

            # Data Preprocessing
            preprocessed_data = preprocess_data_for_customer(data, customer_name, start_date, end_date, frequency)
            imputed_data = impute_data(preprocessed_data, imputation_method)

            # Initialize and fit the AutoTS model
            model = AutoTS(
                forecast_length=int(len(imputed_data) * 0.2),
                frequency=frequency,
                prediction_interval=confidence_interval,
                ensemble='simple',
                max_generations=5,
                num_validations=2,
                validation_method="backwards",
            )
            model = model.fit(imputed_data, date_col='Date', value_col='QTY', id_col=None)

            # Display chosen model
            st.write("Chosen Model by AutoTS:")
            st.text(str(model.best_model))

            # Generate predictions
            prediction = model.predict()
            forecast_df = prediction.forecast

            # Combine forecast and confidence intervals into one DataFrame
            forecast_combined = forecast_df.copy()
            forecast_combined['Forecast Interval'] = forecast_combined.index
            forecast_combined['Lower Confidence Interval'] = prediction.lower_forecast['QTY']
            forecast_combined['Upper Confidence Interval'] = prediction.upper_forecast['QTY']
            forecast_combined.rename(columns={'QTY': 'Forecast Value'}, inplace=True)
            
            # Display forecast with intervals
            st.write("Forecast with Confidence Intervals:")
            st.dataframe(forecast_combined.reset_index(drop=True))

            # Plotting
            fig = plot_data(imputed_data, forecast_combined, forecast_combined['Lower Confidence Interval'], forecast_combined['Upper Confidence Interval'])
            st.pyplot(fig)
