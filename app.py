import streamlit as st
import pandas as pd
from autots import AutoTS
import matplotlib.pyplot as plt
import base64
import numpy as np

# Function to transform customer names
def transform_customer_name(customer_name):
    # Your existing transformation code

# Function to apply custom CSS for background image
def local_css(bg_image):
    # Your existing CSS application code

# Load your image
with open("High_resolution_image_of_wooden_pallets_neatly_sta.png", "rb") as file:
    bg_image = base64.b64encode(file.read()).decode("utf-8")

# Apply the custom CSS
local_css(bg_image)

st.markdown("""
    <h1 style='text-align: center; color: black;'>Demand Forecasting & Optimization of Supply Chain</h1>
    <h2 style='text-align: center; color: black;'>Wooden Pallets</h2>
    """, unsafe_allow_html=True)

st.sidebar.header("Input Options")

# Standard dates for selection
standard_dates = ['2019-01-01', '2021-01-01', '2021-11-01']

# File upload and data preparation
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Customer Name (Cleaned)'] = data['Customer Name'].apply(transform_customer_name)

    # Dropdown for standard date selection
    selected_date = st.sidebar.selectbox("Select From Date", standard_dates)

    # Dropdown for customer selection
    customer_name = st.sidebar.selectbox("Select Customer", data['Customer Name (Cleaned)'].unique())

    # Dropdown for aggregation frequency
    frequency = st.sidebar.selectbox("Select Aggregation Frequency", ['D', 'W', 'M'])

    # Slider for confidence interval
    confidence_interval = st.sidebar.slider("Select Confidence Interval", 0.80, 0.99, 0.95, 0.01)

    # Dropdown for imputation method
    imputation_methods = ['None', 'ffill', 'bfill', 'linear', 'akima', 'cubic']
    imputation_method = st.sidebar.selectbox("Select Imputation Method", imputation_methods)

    # Filter data based on the selected date
    filtered_data = data[data['Date'] >= pd.to_datetime(selected_date)]

    # Filter data based on the selected customer
    customer_filtered_data = filtered_data[filtered_data['Customer Name (Cleaned)'] == customer_name]

    # Resample and aggregate QTY data
    resampled_data = customer_filtered_data.resample(frequency, on='Date')['QTY'].sum().reset_index()

    # Save the original data before imputation
    original_data = resampled_data.copy()

    # Apply imputation if needed
    if imputation_method != 'None':
        resampled_data['QTY'] = resampled_data['QTY'].replace(0, np.nan).interpolate(method=imputation_method)

    # Plot the data before and after imputation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(original_data['Date'], original_data['QTY'], label='Original Data', color='blue')
    ax.plot(resampled_data['Date'], resampled_data['QTY'], label='Imputed Data', color='red', linestyle='--')
    ax.set_title('Data Over Time Before and After Imputation')
    ax.set_xlabel('Date')
    ax.set_ylabel('QTY')
    ax.legend()
    st.pyplot(fig)

    # Forecasting
    if st.sidebar.button("Forecast"):
        with st.spinner('Running the model...'):
            model = AutoTS(
                forecast_length=int(len(resampled_data) * 0.2),
                frequency=frequency,
                prediction_interval=confidence_interval,
                ensemble='simple',
                max_generations=5,
                num_validations=2,
                validation_method="backwards",
            )
            model = model.fit(resampled_data, date_col='Date', value_col='QTY', id_col=None)

            st.write("Chosen Model by AutoTS:")
            try:
                # Retrieve and display the best model's summary
                best_model_summary = model.best_model['Model Summary']
                st.text(best_model_summary)
            except KeyError as e:
                st.error(f"KeyError: {e}")
                st.text(f"Available keys in 'best_model': {list(model.best_model.keys())}")

            prediction = model.predict()
            forecast_df = prediction.forecast
            forecast_combined = forecast_df.copy()
            forecast_combined['Forecast Interval'] = forecast_combined.index
            forecast_combined['Lower Confidence Interval'] = prediction.lower_forecast['QTY']
            forecast_combined['Upper Confidence Interval'] = prediction.upper_forecast['QTY']
            forecast_combined.rename(columns={'QTY': 'Forecast Value'}, inplace=True)

            st.write("Forecast with Confidence Intervals:")
            st.dataframe(forecast_combined.reset_index(drop=True))

            # Plotting the forecast
            plt.figure(figsize=(12, 6))
            plt.plot(original_data['Date'], original_data['QTY'], label='Historical Data', color='blue')
            plt.plot(forecast_combined['Forecast Interval'], forecast_combined['Forecast Value'], label='Forecasted Data', color='green', linestyle='--')
            plt.fill_between(forecast_combined['Forecast Interval'], forecast_combined['Lower Confidence Interval'], forecast_combined['Upper Confidence Interval'], color='gray', alpha=0.3, label='Confidence Interval')
            plt.title('Historical vs Forecasted Data with Confidence Intervals')
            plt.xlabel('Date')
            plt.ylabel('QTY')
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
