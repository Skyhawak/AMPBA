import streamlit as st
import pandas as pd
from autots import AutoTS
import matplotlib.pyplot as plt
import base64
import numpy as np

# Function to apply custom CSS for background image
def local_css(bg_image):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load your image
with open("High_resolution_image_of_wooden_pallets_neatly_sta.png", "rb") as file:
    bg_image = base64.b64encode(file.read()).decode("utf-8")

# Apply the custom CSS
local_css(bg_image)

st.markdown("""
    <h1 style='text-align: center; color: black; margin-bottom: 0px;'>Demand Forecasting & Optimization of Supply Chain</h1>
    <h2 style='text-align: center; color: black; margin-top: 0px;'>Wooden Pallets</h2>
    """, unsafe_allow_html=True)

st.sidebar.header("Input Options")

# Standard dates for data processing
standard_dates = [pd.to_datetime('2019-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-11-01')]

# Function to preprocess data for a specific customer
def preprocess_data_for_customer(data, customer_name, frequency):
    customer_data = data[data['Customer Name (Cleaned)'] == customer_name]
    processed_data = pd.DataFrame()
    for sd in standard_dates:
        date_filtered_data = customer_data[customer_data['Date'] >= sd]
        # Ensure only relevant columns are aggregated
        cols_to_aggregate = date_filtered_data[['Date', 'QTY']]
        aggregated_data = cols_to_aggregate.resample(frequency, on='Date').sum().reset_index()
        processed_data = pd.concat([processed_data, aggregated_data])
    return processed_data

# Function to plot data
def plot_combined_data(original_data, imputed_data):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data['Date'], original_data['QTY'], label='Data Before Imputation', color='blue')
    if imputed_data is not None:
        plt.plot(imputed_data['Date'], imputed_data['QTY'], label='Data After Imputation', color='red', linestyle='--')
    plt.title('Data Over Time Before and After Imputation')
    plt.xlabel('Date')
    plt.ylabel('QTY')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# File upload
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "csv"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])

    customer_name = st.sidebar.selectbox("Select Customer", data['Customer Name (Cleaned)'].unique())
    frequency = st.sidebar.selectbox("Select Aggregation Frequency", ['15D', 'W', 'M'])
    confidence_interval = st.sidebar.slider("Select Confidence Interval", 0.80, 0.99, 0.95, 0.01)

    preprocessed_data = preprocess_data_for_customer(data, customer_name, frequency)
    
    # Imputation selection
    imputation_methods = ['None', 'ffill', 'bfill', 'linear', 'akima', 'cubic']
    imputation_method = st.sidebar.selectbox("Select Imputation Method", imputation_methods)
    imputed_data = None
    if imputation_method != 'None':
        imputed_data = preprocessed_data.copy()
        imputed_data['QTY'] = imputed_data['QTY'].replace(0, np.nan)
        if imputation_method in imputation_methods[1:]:  # Check if the method is in the list
            imputed_data['QTY'] = imputed_data['QTY'].interpolate(method=imputation_method)

    # Plot combined data
    st.subheader("Data Over Time Before and After Imputation")
    plot_combined_data(preprocessed_data, imputed_data)

    # Forecasting and plotting
    if st.sidebar.button("Forecast"):
        with st.spinner('Running the model...'):
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
            plt.plot(imputed_data['Date'], imputed_data['QTY'], label='Historical Data', color='blue')
            plt.plot(forecast_combined['Forecast Interval'], forecast_combined['Forecast Value'], label='Forecasted Data', color='green', linestyle='--')
            plt.fill_between(forecast_combined['Forecast Interval'], forecast_combined['Lower Confidence Interval'], forecast_combined['Upper Confidence Interval'], color='gray', alpha=0.3, label='Confidence Interval')
            plt.title('Historical vs Forecasted Data with Confidence Intervals')
            plt.xlabel('Date')
    plt.ylabel('QTY')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
