import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import numpy as np

# Function to transform customer names
def transform_customer_name(customer_name):
    customer_name = customer_name.strip()
    customer_name = ' '.join(customer_name.split())
    return customer_name

# Function to apply custom CSS for background image
def local_css(bg_image):
    bg_image = base64.b64encode(bg_image.read()).decode("utf-8")
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

# Apply the custom CSS
with open("High_resolution_image_of_wooden_pallets_neatly_sta.png", "rb") as file:
    local_css(file)

st.markdown("""
    <h1 style='text-align: center; color: black; margin-bottom: 0px;'>Demand Forecasting & Optimization of Supply Chain</h1>
    <h2 style='text-align: center; color: black; margin-top: 0px;'>Wooden Pallets</h2>
    """, unsafe_allow_html=True)

st.sidebar.header("Input Options")

# Standard dates for data processing
standard_dates = ['2019-01-01', '2021-01-01', '2021-11-01']
date_selection = st.sidebar.selectbox("Select From Date", standard_dates)

# File upload and data preparation
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Customer Name (Cleaned)'] = data['Customer Name'].apply(transform_customer_name)

    # Select customer, frequency, confidence interval, and imputation method
    customer_name = st.sidebar.selectbox("Select Customer", data['Customer Name (Cleaned)'].unique())
    frequency = st.sidebar.selectbox("Select Aggregation Frequency", ['15D', 'W', 'M'])
    confidence_interval = st.sidebar.slider("Select Confidence Interval", 0.80, 0.99, 0.95, 0.01)
    imputation_methods = ['None', 'ffill', 'bfill', 'linear', 'akima', 'cubic']
    imputation_method = st.sidebar.selectbox("Select Imputation Method", imputation_methods)

    # Filter data based on selected customer and date
    customer_data = data[data['Customer Name (Cleaned)'] == customer_name]
    customer_data = customer_data[customer_data['Date'] >= pd.to_datetime(date_selection)]

    # Resample and aggregate QTY data
    resampled_data = customer_data.resample(frequency, on='Date')['QTY'].sum().reset_index()
    
    # Impute if necessary
    if imputation_method != 'None':
        resampled_data['QTY'] = resampled_data['QTY'].replace(0, np.nan)
        resampled_data['QTY'] = resampled_data['QTY'].interpolate(method=imputation_method)

    # Plotting the data
    fig, ax = plt.subplots()
    ax.plot(resampled_data['Date'], resampled_data['QTY'], label='Data After Imputation', color='red', linestyle='--')
    ax.set_title('Data Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('QTY')
    ax.legend()
    st.pyplot(fig)
