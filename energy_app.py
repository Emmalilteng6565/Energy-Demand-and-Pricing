import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import load


countries = ['Austria', 'Belgium', 'Bulgaria', 'Switzerland', 'Czechia',
       'Germany', 'Denmark', 'Spain', 'Estonia', 'Finland', 'France',
       'Greece', 'Croatia', 'Hungary', 'Ireland', 'Italy', 'Lithuania',
       'Luxembourg', 'Latvia', 'North Macedonia', 'Netherlands', 'Norway',
       'Poland', 'Portugal', 'Romania', 'Serbia', 'Slovakia', 'Slovenia',
       'Sweden']

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
    year = st.selectbox('Year', list(range(2000, 2024)))
    month = st.selectbox('Month', list(range(1, 13)))
    day = st.slider('Day', 1, 31)
    country = st.selectbox('Country', countries)

    if st.button('Predict'):
        # Prepare the feature vector for prediction
        features = np.array([[year, month,day,country]])

        # Make prediction
        prediction = model.predict(features)

        # Display prediction
        st.write(f'The predicted energy price for{country} for the date {month}/{year} is {prediction[0]}')
