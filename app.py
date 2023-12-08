import streamlit as st
import pandas as pd
from autots import AutoTS
import matplotlib.pyplot as plt
import base64
import numpy as np

# Function to apply custom CSS for background image
def local_css():
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
local_css()

st.markdown("""
    <h1 style='text-align: center; color: black;'>Demand Forecasting & Optimization of Supply Chain</h1>
    <h2 style='text-align: center; font-size: 24px;'><i>Wooden Pallets</i></h2>
    """, unsafe_allow_html=True)
st.markdown('<p style="font-size: 24px;"><i>Wooden Pallets</i></p>', unsafe_allow_html=True)

st.sidebar.header("Input Options")

# Function to preprocess data for a specific customer
def preprocess_data_for_customer(data, customer_name, start_date, end_date, frequency):
    customer_data = data[data['Customer Name (Cleaned)'] == customer_name]
    customer_data = customer_data[(customer_data['Date'] >= start_date) & (customer_data['Date'] <= end_date)]
    customer_data = customer_data.groupby('Date').agg({'QTY': 'sum'}).reset_index()
    aggregated_data = customer_data.set_index('Date').resample(frequency).sum().reset_index()
    return aggregated_data

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
    data = pd.read_csv(uploaded_file) if uploaded_file.type == "text/csv" else pd.read_excel(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    customer_name = st.sidebar.selectbox("Select Customer", data['Customer Name (Cleaned)'].unique())
    frequency = st.sidebar.selectbox("Select Aggregation Frequency", ['15D', 'W', 'M'])

    preprocessed_data = preprocess_data_for_customer(data, customer_name, data['Date'].min(), data['Date'].max(), frequency)
    
    # Imputation selection
    imputation_method = st.sidebar.selectbox("Select Imputation Method", ['None', 'ffill', 'bfill', 'linear'])
    imputed_data = None
    if imputation_method != 'None':
        imputed_data = preprocessed_data.copy()
        imputed_data['QTY'] = imputed_data['QTY'].replace(0, np.nan)
        if imputation_method == 'ffill':
            imputed_data['QTY'] = imputed_data['QTY'].fillna(method='ffill')
        elif imputation_method == 'bfill':
            imputed_data['QTY'] = imputed_data['QTY'].fillna(method='bfill')
        else:
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
                prediction_interval=0.9,  # Assuming a fixed confidence interval
                ensemble='simple',
                max_generations=5,
                num_validations=2,
                validation_method="backwards",
            )
            model = model.fit(imputed_data, date_col='Date', value_col='QTY', id_col=None)

            st.write("Chosen Model by AutoTS:")
            st.text(str(model.best_model))  # Convert the model details to a string

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
