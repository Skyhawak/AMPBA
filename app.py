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
standard_dates = {
    '2019-01-01': pd.to_datetime('2019-01-01'),
    '2021-01-01': pd.to_datetime('2021-01-01'),
    '2021-11-01': pd.to_datetime('2021-11-01')
}

# Date selection
selected_date = st.sidebar.selectbox("Select From Date", options=list(standard_dates.keys()))

# File upload
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file) if uploaded_file.type == "text/csv" else pd.read_excel(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter data from the selected start date to the last data point
    filtered_data = data[data['Date'] >= standard_dates[selected_date]]

    customer_name = st.sidebar.selectbox("Select Customer", filtered_data['Customer Name'].unique())
    frequency = st.sidebar.selectbox("Select Aggregation Frequency", ['15D', 'W', 'M'])
    confidence_interval = st.sidebar.slider("Select Confidence Interval", 0.80, 0.99, 0.95, 0.01)

    # Aggregating data
    aggregated_data = filtered_data.set_index('Date').resample(frequency).sum().reset_index()
    aggregated_data_before_imputation = aggregated_data.copy()

    # Imputation selection
    imputation_methods = ['None', 'ffill', 'bfill', 'linear', 'akima', 'cubic']
    imputation_method = st.sidebar.selectbox("Select Imputation Method", imputation_methods)
    if imputation_method != 'None':
        aggregated_data['QTY'] = aggregated_data['QTY'].replace(0, np.nan).interpolate(method=imputation_method)

    # Plot combined data
    st.subheader("Data Over Time Before and After Imputation")
    plt.figure(figsize=(12, 6))
    plt.plot(aggregated_data_before_imputation['Date'], aggregated_data_before_imputation['QTY'], label='Data Before Imputation', color='blue')
    plt.plot(aggregated_data['Date'], aggregated_data['QTY'], label='Data After Imputation', color='red', linestyle='--')
    plt.title('Data Over Time Before and After Imputation')
    plt.xlabel('Date')
    plt.ylabel('QTY')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

    # Forecasting and plotting if the forecast button is clicked
    if st.sidebar.button("Forecast"):
        with st.spinner('Running the model...'):
            model = AutoTS(
                forecast_length=int(len(aggregated_data) * 0.2),
                frequency=frequency,
                prediction_interval=confidence_interval,
                ensemble='simple',
                max_generations=5,
                num_validations=2,
                validation_method="backwards",
            )
            model = model.fit(aggregated_data, date_col='Date', value_col='QTY', id_col=None)

            st.write("Chosen Model by AutoTS:")
            try:
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
