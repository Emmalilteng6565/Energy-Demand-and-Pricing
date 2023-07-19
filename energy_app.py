import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained models
gb_demand_model = joblib.load('gb_demand_model.joblib')
gb_price_model = joblib.load('gb_price_model.joblib')

# Load the Scaler
scaler = joblib.load('scaler.joblib')

# Define the Streamlit app
def main():
    # Title
    st.title("Energy Demand and Price Prediction")

    # Input features
    rainfall = st.slider('Rainfall', min_value=0.0, max_value=50.0, step=0.1)
    solar_exposure = st.slider('Solar Exposure', min_value=0.0, max_value=50.0, step=0.1)
    school_day = st.selectbox('School Day (1: Yes, 0: No)', options=[1, 0])
    holiday = st.selectbox('Holiday (1: Yes, 0: No)', options=[1, 0])
    day = st.selectbox('Day of the Month', options=list(range(1,32)))
    month = st.selectbox('Month', options=list(range(1,13)))
    year = st.selectbox('Year', options=list(range(2010, 2024)))

    # Create a dataframe from the inputs
    input_data = {'rainfall': [rainfall], 'solar_exposure': [solar_exposure], 'school_day': [school_day], 
                  'holiday': [holiday], 'day': [day], 'month': [month], 'year': [year]}
    input_df = pd.DataFrame(input_data)

    # Scale the input data
    input_df_scaled = scaler.transform(input_df)

    # Make a demand prediction
    demand_prediction = gb_demand_model.predict(input_df_scaled)[0]
    st.write(f"Predicted Demand: {demand_prediction} MWh")

    # Add the demand prediction to the input data
    input_df['demand'] = [demand_prediction]

    # Scale the input data with the demand prediction
    input_df_scaled = scaler.transform(input_df)

    # Make a price prediction
    price_prediction = gb_price_model.predict(input_df_scaled)[0]
    st.write(f"Predicted Price: {price_prediction} AUD/MWh")

if __name__ == "__main__":
    main()
