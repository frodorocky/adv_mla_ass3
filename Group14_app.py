#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:19:16 2023

@author: frodo
"""

import streamlit as st
import pandas as pd
import joblib
import gdown 
from src.data.unzip_merge_csv import unzip_and_merge_csvs
from src.features.preprocessor import DataFramePreprocessor

# Function to load and preprocess data (cached to prevent reloading on each interaction)
@st.cache_data
def load_and_preprocess_data(base_folder_path):
    df = unzip_and_merge_csvs(base_folder_path)
    preprocessor = DataFramePreprocessor(df.copy())
    return preprocessor.preprocess()

# Function to load a specific model (not cached)
def load_model(model_name):
    if model_name == 'lightgbm':
        return joblib.load('models/lightgbm_pipeline.pkl')
    elif model_name == 'decision_tree':
        return joblib.load('models/decision_tree_regression.pkl')
    elif model_name == 'random_forest':
        return joblib.load('models/rf_regression.pkl')
    elif model_name == 'xgboost':
        return joblib.load('models/xgb_pipeline.pkl')
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def load_data_in_chunks(file_path, chunk_size=50000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, axis=0)

# Load data
base_folder_path = 'data/raw/itineraries_csv'
df = load_and_preprocess_data(base_folder_path)

# Streamlit application interface
st.title('Flight Fare Prediction')

# User inputs
origin_airport = st.text_input("Origin airport")
destination_airport = st.text_input("Destination airport")
departure_date = st.date_input("Departure date")
departure_time = st.time_input("Departure time")
cabin_type = st.selectbox("Cabin type", df['segmentsCabinCode'].unique())

# Prediction function
def predict_fare(model_pipeline, X):
    return model_pipeline.predict(X)

# Prepare data for prediction
def get_filtered_data():
    return df[
        (df['startingAirport'] == origin_airport) &
        (df['destinationAirport'] == destination_airport) &
        (df['flightDate'] == str(departure_date)) &
        (df['departure_hour'] == departure_time.hour) &
        (df['segmentsCabinCode'] == cabin_type)
    ]

# Predict and display fare for the chosen model
def display_prediction(model_name):
    model_pipeline = load_model(model_name)  # Load the model when the button is clicked
    filtered_df = get_filtered_data()
    
    if not filtered_df.empty:
        filtered_df = filtered_df.set_index(['legId'])
        X = filtered_df.drop('totalFare', axis=1)
        predicted_fare = predict_fare(model_pipeline, X)
        st.write(f"Predicted Fare by {model_name}: ${predicted_fare[0]:.2f}")
    else:
        st.write("No matching historical data found. Unable to predict fare.")

# Buttons for prediction with each model
if st.button('Predict Fare with LightGBM'):
    display_prediction('lightgbm')

if st.button('Predict Fare with Random Forest'):
    display_prediction('random_forest')

if st.button('Predict Fare with Decision Tree'):
    display_prediction('decision_tree')

if st.button('Predict Fare with XGBoost'):
    display_prediction('xgboost')
