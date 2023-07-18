import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import load


countries = [ 57.54058256,  46.69267472,  71.52281789,  73.93226211,
        91.81232862,  90.00960067, 104.46860823,  67.73344955,
        84.24664738,  69.40521539,  53.74111835,  66.41854184,
        74.53456737,  96.99191778,  72.58268036,  72.58346566,
        75.16897192,  77.15773591,  82.26388317,  83.90998991,
        75.84687003,  65.69469277,  81.691089  ,  67.22618391,
        69.77377764,  49.17303124,  77.85180908,  91.67756288]

# Function to load the trained model
def load_model(path):
    try:
        return load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the trained model
model = load_model('model.joblib')

if model is not None:
    st.title('Energy Price Prediction App')

    # User input
    year = st.selectbox('Year', list(range(2022, 2023, 2024, 2025)))
    month = st.selectbox('Month', list(range(1, 13)))
    day = st.slider('Day', 1, 31)
    country = st.selectbox('Country', countries)

    if st.button('Predict'):
        # Prepare the feature vector for prediction
        features = np.array([[year, month,day,country]])

        # Make prediction
        prediction = model.predict(features)

        # Display prediction
        st.write(f'The predicted energy price for {country} for the date {day}/{month}/{year} is ${prediction[0]}')
