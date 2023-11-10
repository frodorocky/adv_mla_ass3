#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:01:40 2023

@author: frodo
"""

import streamlit as st
import pandas as pd
import joblib
import os
from src.data.unzip_merge_csv import unzip_and_merge_csvs
from src.features.preprocessor import DataFramePreprocessor

# Load and preprocess data (cached to prevent reloading on each interaction)
@st.cache_data
def load_and_preprocess_data(base_folder_path):
    # Load data from CSVs and merge
    df = unzip_and_merge_csvs(base_folder_path)
    # Initialize and apply the data preprocessor
    preprocessor = DataFramePreprocessor(df.copy())
    return preprocessor.preprocess()

# Load models (cached to prevent reloading on each interaction)
@st.cache_resource
def load_models():
    lightgbm_pipeline = joblib.load('models/lightgbm_pipeline.pkl')
    decisiontree_pipeline = joblib.load('models/decision_tree_regression.pkl')
    rf_pipeline = joblib.load('models/rf_regression.pkl')
    #xgb_pipeline = joblib.load('models/xgb.pkl')
    # Return all the loaded models
    return lightgbm_pipeline, decisiontree_pipeline, rf_pipeline, #xgb_pipeline

# Data and model loading
base_folder_path = 'data/raw/itineraries_csv'
df = load_and_preprocess_data(base_folder_path)
lightgbm_pipeline, decisiontree_pipeline, rf_pipeline = load_models()

# Streamlit application interface
st.title('Flight Fare Prediction')

# User inputs
origin_airport = st.text_input("Origin airport")
destination_airport = st.text_input("Destination airport")
departure_date = st.date_input("Departure date")
departure_time = st.time_input("Departure time")
cabin_type = st.selectbox("Cabin type", df['segmentsCabinCode'].unique())

# Prediction functions
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

# Predict and display fare for each model
def display_prediction(model_pipeline, model_name):
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
    display_prediction(lightgbm_pipeline, 'LightGBM')

if st.button('Predict Fare with Random Forest'):
    display_prediction(rf_pipeline, 'Random Forest')

if st.button('Predict Fare with Decision Tree'):
    display_prediction(decisiontree_pipeline, 'decision_tree')

if st.button('Predict Fare with XGBoost'):
    display_prediction(xgb_pipeline, 'XGBoost')