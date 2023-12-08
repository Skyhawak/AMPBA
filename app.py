import streamlit as st
import pandas as pd
import plotly.express as px
from autots import AutoTS
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

st.title("Wooden Pallets Demand Forecasting")

st.sidebar.header("Input Options")

# Function to preprocess data for a specific customer
def preprocess_data_for_customer(data, customer_name, start_date, end_date, frequency):
    customer_data = data[data['Customer Name (Cleaned)'] == customer_name]
    customer_data = customer_data[(customer_data['Date'] >= start_date) & (customer_data['Date'] <= end_date)]
    customer_data = customer_data.groupby('Date').agg({'QTY': 'sum'}).reset_index()
    aggregated_data = customer_data.set_index('Date').resample(frequency).sum().reset_index()
    return aggregated_data

# Function to plot data with Plotly
def plot_combined_data_plotly(original_data, imputed_data=None):
    fig = px.line(original_data, x='Date', y='QTY', title='Data Over Time', labels={'QTY': 'Quantity'}, markers=True)
    if imputed_data is not None:
        fig.add_scatter(x=imputed_data['Date'], y=imputed_data['QTY'], mode='lines+markers', name='Imputed Data', line=dict(dash='dash'))
    fig.update_layout(title='Data Over Time Before and After Imputation')
    st.plotly_chart(fig, use_container_width=True)

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
    plot_combined_data_plotly(preprocessed_data, imputed_data)

    # Forecasting
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
            forecast_fig = px.line(forecast_combined, x='Forecast Interval', y='Forecast Value', title='Forecasted Data', markers=True)
            forecast_fig.add_scatter(x=forecast_combined['Forecast Interval'], y=forecast_combined['Lower Confidence Interval'], fill='tonexty', mode='lines', line=dict(color="lightgrey"), name='Lower Confidence')
            forecast_fig.add_scatter(x=forecast_combined['Forecast Interval'], y=forecast_combined['Upper Confidence Interval'], fill='tonexty', mode='lines', line=dict(color="lightgrey"), name='Upper Confidence')
            forecast_fig.update_layout(title='Historical vs Forecasted Data with Confidence Intervals')
            st.plotly_chart(forecast_fig, use_container_width=True)
