import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import json 
from autots import AutoTS

# Function to transform customer names
def transform_customer_name(customer_name):
    customer_name = customer_name.strip()
    customer_name = ' '.join(customer_name.split())
    customer_name = customer_name.replace("-", "_")
    customer_name = customer_name.replace("Pvt. Ltd.", "Private Limited")
    customer_name = customer_name.replace("Pvt Ltd", "Private Limited")
    customer_name = customer_name.replace("Pvt. Limited", "Private Limited")
    customer_name = customer_name.replace(" _", "_").replace("_ ", "_")
    return customer_name

# Function to apply custom CSS for background image
def local_css(file_path):
    with open(file_path, "rb") as file:
        bg_image = base64.b64encode(file.read()).decode("utf-8")
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
local_css("High_resolution_image_of_wooden_pallets_neatly_sta.png")  # Make sure to replace with the correct path

st.markdown("""
    <h1 style='text-align: center; color: black;'>Demand Forecasting & Optimization of Supply Chain</h1>
    <h2 style='text-align: center; color: black;'>Wooden Pallets</h2>
    """, unsafe_allow_html=True)

st.sidebar.header("Input Options")

# File upload and data preparation
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    df['Customer Name (Cleaned)'] = df['Customer Name'].apply(transform_customer_name)
    df['Date'] = pd.to_datetime(df['Date'])

    # Specific customer names to include
    customer_names_to_include = ['Alpla India Private Limited_Sangareddy', 'Babri Polypet Private Limited_Haridwar', 'Bericap India Private Limited_Pune',
                             'Chemco Plastic Industries Private Limited_Vadodara','Epitome Petropack Limited_Kolkata','Hindustan Coca Cola Beverages Private Limited_Goblej Plant_HMA1',
                             'Hindustan Coca Cola Beverages Private Limited_Bidadi Greenfield_HMKJ','Hindustan Coca Cola Beverages Private Limited_Bidadi Old Plant_HMK1','Hindustan Coca Cola Beverages Private Limited_Ahmedabad Sanand_HMAH',
                             'Hindustan Coca Cola Beverages Private Limited_Khurda_HMF1','Hindustan Coca Cola Beverages Private Limited_Kanchenkanya_HMS3','Hindustan Coca Cola Beverages Private Limited_GAPL_Howrah','Manjushree Technopack Limited_Bangalore_Bidadi',
                             'Oricon Enterprises Limited_Khordha','Pepsico India Holdings Private Limited_Patiala_Channo','SLMG Beverages Private Limited_Lucknow']  # Your list of customer names

    # Filter the data for specific customers and 'Model 2' == 'Allot'
    df_filtered = df[(df['Customer Name (Cleaned)'].isin(customer_names_to_include)) & (df['Model 2'] == 'Allot')]

    # Dropdown for customer selection
    customer_name = st.sidebar.selectbox("Select Customer", df_filtered['Customer Name (Cleaned)'].unique())

    if customer_name:
        # Standard dates for selection
        standard_dates = ['2019-01-01', '2021-01-01', '2021-11-01']
        selected_date = st.sidebar.selectbox("Select From Date", standard_dates)

        if selected_date:
            # Further filter the DataFrame from the selected date onwards
            final_filtered_data = df_filtered[df_filtered['Date'] >= pd.to_datetime(selected_date)]

            # Dropdown for aggregation frequency
            frequency = st.sidebar.selectbox("Select Aggregation Frequency", ['15D', 'W', 'M'])

            # Slider for confidence interval
            confidence_interval = st.sidebar.slider("Select Confidence Interval", 0.80, 0.99, 0.95, 0.01)

            # Dropdown for imputation method
            imputation_methods = ['None', 'ffill', 'bfill', 'linear', 'akima', 'cubic']
            imputation_method = st.sidebar.selectbox("Select Imputation Method", imputation_methods)

            # Resample and aggregate QTY data
            resampled_data = final_filtered_data.resample(frequency, on='Date')['QTY'].sum().reset_index()

            # Save the original data before imputation
            original_data = resampled_data.copy()

           
            # Apply imputation if needed
            if imputation_method != 'None':
                resampled_data['QTY'].replace(0, np.nan, inplace=True)
                resampled_data['QTY'] = resampled_data['QTY'].interpolate(method=imputation_method)

            # Plot the data before and after imputation
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(original_data['Date'], original_data['QTY'], label='Original Data', color='blue')
            if imputation_method != 'None':
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
                        forecast_length=20,  # Forecasting 20 future values
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
                        best_model_summary = model.best_model['Model Summary'] if 'Model Summary' in model.best_model else "No 'Model Summary' found"
                        st.text(best_model_summary)
                    except KeyError as e:
                        st.error(f"KeyError: {e}")

                    prediction = model.predict()
                    forecast_df = prediction.forecast.round(0).astype(int)  # Ensure forecast values are integers
                    forecast_combined = forecast_df.copy()
                    forecast_combined['Forecast Interval'] = forecast_combined.index
                    forecast_combined['Lower Confidence Interval'] = prediction.lower_forecast['QTY']
                    forecast_combined['Upper Confidence Interval'] = prediction.upper_forecast['QTY']
                    forecast_combined.rename(columns={'QTY': 'Forecast Value'}, inplace=True)

                    st.write("Forecast with Confidence Intervals:")
                    st.dataframe(forecast_combined.reset_index(drop=True))

                    # Plotting the forecast
                    forecast_fig, forecast_ax = plt.subplots(figsize=(12, 6))
                    forecast_ax.plot(resampled_data['Date'], resampled_data['QTY'], label='Historical Data', color='blue')
                    forecast_ax.plot(forecast_combined['Forecast Interval'], forecast_combined['Forecast Value'], label='Forecasted Data', color='green', linestyle='--')
                    forecast_ax.fill_between(forecast_combined['Forecast Interval'], forecast_combined['Lower Confidence Interval'], forecast_combined['Upper Confidence Interval'], color='gray', alpha=0.3, label='Confidence Interval')
                    forecast_ax.set_title('Historical vs Forecasted Data with Confidence Intervals')
                    forecast_ax.set_xlabel('Date')
                    forecast_ax.set_ylabel('QTY')
                    forecast_ax.legend()
                    plt.tight_layout()
                    st.pyplot(forecast_fig)
            
          
