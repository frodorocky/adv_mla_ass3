import os
import zipfile
import pandas as pd
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

pd.options.display.max_columns = None
pd.set_option('display.max_rows', 100)

# Load data
dataframes = []

# Extract csv files from zip files
for root, dirs, files in os.walk("itineraries_csv"):
    for file in files:
        if file.endswith(".zip"):
            zip_file_path = os.path.join(root, file)

            # Open and extract the ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                for zip_info in zip_file.infolist():
                    if zip_info.filename.endswith(".csv"):
                        with zip_file.open(zip_info) as csv_file:
                            # Read the CSV data into a Pandas DataFrame
                            df = pd.read_csv(csv_file)
                            dataframes.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.info()

df = pd.read_csv('c:/Users/brand/OneDrive/Desktop/ATL_itineraries_aa.csv')

d = joblib.load('C:/Users/brand/OneDrive/Documents/GitHub/AT3-MACHINELEARNING/xgb.pkl')    

# Drop legID
cleaned_df = combined_df
cleaned_df = combined_df.drop(columns = 'legId')

# Convert column data types
cleaned_df['searchDate'] = pd.to_datetime(combined_df['searchDate'])
cleaned_df['flightDate'] = pd.to_datetime(combined_df['flightDate'])

# Convert travelDuration to minutes
def convert_travelduration(duration):
    matches = re.findall(r'PT(\d+)H(\d+)M', duration)
    if matches:
        hours, minutes = map(int, matches[0])
        return hours * 60 + minutes
    else:
        return 0

cleaned_df['travelDuration'] = cleaned_df['travelDuration'].apply(convert_travelduration)

# Extract segments
extracted_df = cleaned_df

def extract_segment_arrival(s):
    segments = s.split("||")
    return segments[-1]

extracted_df['ArrivalTime'] = extracted_df['segmentsArrivalTimeRaw'].apply(extract_segment_arrival)

def extract_segment_departure(s):
    segments = s.split("||")
    return segments[0]

extracted_df['DepartureTime'] = extracted_df['segmentsDepartureTimeRaw'].apply(extract_segment_departure)

#Check null values
extracted_df = joblib.load('extracted.joblib')
removed_null = extracted_df.dropna()
len(removed_null)
len(extracted_df)

#Drop cols
removed_null = removed_null.drop(columns = 'segmentsDepartureTimeRaw')
removed_null = removed_null.drop(columns = 'segmentsArrivalTimeRaw')
removed_null = removed_null.drop(columns = 'segmentsAirlineName')
removed_null = removed_null.drop(columns = 'segmentsDepartureTimeEpochSeconds')
removed_null = removed_null.drop(columns = 'segmentsArrivalTimeEpochSeconds')
removed_null = removed_null.drop(columns = 'segmentsDurationInSeconds')

#Create stopnnumber col
stopnumber = []
for index, value in enumerate(removed_null['segmentsDepartureAirportCode']):
    if "||" in value:
        segments = value.split("||")
        count = len(segments) - 1
        stopnumber.append(count)
    else:
        count = 0
        stopnumber.append(count)

removed_null['stopnumber'] = stopnumber

#Drop segmentsDepartureAirportCode and segmentsArrivalAirportCode
removed_null = removed_null.drop(columns = 'segmentsArrivalAirportCode')
removed_null = removed_null.drop(columns = 'segmentsDepartureAirportCode')

#Drop segmentsDistance
removed_null = removed_null.drop(columns = 'segmentsDistance')

#Create new cols for cabin type
removed_null['iscoach'] = removed_null['segmentsCabinCode'].apply(
    lambda x: any(substring in x.split("||") for substring in ['coach']))

removed_null['ispremiumcoach'] = removed_null['segmentsCabinCode'].apply(
    lambda x: any(substring in x.split("||") for substring in ['premium coach']))

removed_null['isfirst'] = removed_null['segmentsCabinCode'].apply(
    lambda x: any(substring in x.split("||") for substring in ['first']))

removed_null['isbusiness'] = removed_null['segmentsCabinCode'].apply(
    lambda x: any(substring in x.split("||") for substring in ['business']))

removed_null = removed_null.drop(columns = 'segmentsCabinCode')

#Drop segmentsequipmentdescription, search date
removed_null = removed_null.drop(columns = 'segmentsEquipmentDescription')
removed_null = removed_null.drop(columns = 'searchDate')

#Only keep first airline code
first = []
for index, value in enumerate(removed_null['segmentsAirlineCode']):
    segments = value.split("||")
    s = segments[0]
    first.append(s)

removed_null['airline_code']  = first
removed_null = removed_null.drop(columns ='segmentsAirlineCode')

#Split date cols into month,day,hour,minute
removed_null['DepartureTime'] = pd.to_datetime(removed_null['DepartureTime'], utc=True)
removed_null['ArrivalTime'] = pd.to_datetime(removed_null['ArrivalTime'], utc=True)

removed_null['departuremonth'] = removed_null['DepartureTime'].dt.month
removed_null['departureday'] = removed_null['DepartureTime'].dt.day
removed_null['departurehour'] = removed_null['DepartureTime'].dt.hour
removed_null['departureminute'] = removed_null['DepartureTime'].dt.minute

removed_null['arrivalmonth'] = removed_null['ArrivalTime'].dt.month
removed_null['arrivalday'] = removed_null['ArrivalTime'].dt.day
removed_null['arrivalhour'] = removed_null['ArrivalTime'].dt.hour
removed_null['arrivalminute'] = removed_null['ArrivalTime'].dt.minute

removed_null = removed_null.drop(columns ='flightDate')
removed_null = removed_null.drop(columns ='DepartureTime')
removed_null = removed_null.drop(columns ='ArrivalTime')

removed_null.info(5)
removed_null.head(10)

# Change categorical variables to numeric
removed_null['isBasicEconomy'] = removed_null['isBasicEconomy'].astype(int)
removed_null['isRefundable']= removed_null['isRefundable'].astype(int)
removed_null['isNonStop']= removed_null['isNonStop'].astype(int)
removed_null['iscoach']= removed_null['iscoach'].astype(int)
removed_null['ispremiumcoach']= removed_null['ispremiumcoach'].astype(int)
removed_null['isbusiness']= removed_null['isbusiness'].astype(int)
removed_null['isfirst']= removed_null['isfirst'].astype(int)
removed_null['startairportlabel'] = label_encoder.fit_transform(removed_null['startingAirport'])
removed_null['destinationairportlabel'] = label_encoder.fit_transform(removed_null['destinationAirport'])
removed_null['airlinecodelabel'] = label_encoder.fit_transform(removed_null['airline_code'])


#Modelling
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV

features = removed_null.drop(columns = ['totalFare', 'airline_code', 'destinationAirport', 'startingAirport'])
features.info()
target = removed_null['totalFare']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f'rmse is {rmse}')

params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}

xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='neg_mean_squared_error', 
                   verbose=1)

clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
#Best parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 1000}Lowest RMSE:  74.82035882426383

#Optimised model
model = xgb.XGBRegressor(colsample_bytree = 0.7, learning_rate = 0.1, max_depth = 10, n_estimators =1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f'rmse is {rmse}')

#Create pipeline for streamlit app
import numpy as np

def preprocessor(df):
    
    def cabin_type(df,col):
        df['iscoach'] = df[col].apply(
            lambda x: any(substring in x.split("||") for substring in ['coach']))

        df['ispremiumcoach'] = df[col].apply(
            lambda x: any(substring in x.split("||") for substring in ['premium coach']))

        df['isfirst'] = df[col].apply(
            lambda x: any(substring in x.split("||") for substring in ['first']))

        df['isbusiness'] = df[col].apply(
            lambda x: any(substring in x.split("||") for substring in ['business']))
        return  df['iscoach'],  df['ispremiumcoach'],  df['isfirst'], df['isbusiness']

    def split_departure(df,col):
        date = pd.to_datetime(df[col])
        df['departuremonth'] = date.dt.month
        df['departureday'] = date.dt.day
        return df['departuremonth'], df['departureday']

    def label_encoder(df,col1, col2):
        start = joblib.load('label_encoder_start.joblib')
        end = joblib.load('label_encoder_end.joblib')
        df['startingAirport'] =  start.transform(df[col1])
        df['destinationAirport'] =  end.transform(df[col2])
        return df['startingAirport'], df['destinationAirport']
    
    #Fill df with model features
    df['travelDuration'] = np.nan
    df['isBasicEconomy'] = np.nan
    df['isRefundable'] = np.nan
    df['isNonStop'] = np.nan
    df['totalTravelDistance'] = np.nan
    df['stopnumber'] = np.nan
    df['arrivalmonth'] = np.nan
    df['arrivalday'] = np.nan
    df['arrivalhour'] = np.nan
    df['arrivalminute'] = np.nan
    df['airlinecodelabel'] = np.nan
    date = split_departure(df,'flightDate')
    df['departuremonth'] = date[0]
    df['departureday'] = date[1]
    df['departurehour'] = df['departure_hour']
    df['departureminute'] = np.nan
    cabins = cabin_type(df,col = 'segmentsCabinCode')
    df['iscoach'] = cabins[0].astype(int)
    df['ispremiumcoach'] =cabins[1].astype(int)
    df['isfirst'] =cabins[2].astype(int)
    df['isbusiness'] = cabins[3].astype(int)
    labels = label_encoder(df,col1 = 'startingAirport', col2='destinationAirport')
    df['startairportlabel'] = labels[0]
    df['destinationairportlabel'] = labels[1]
    df = df.drop('flightDate', axis=1)
    df = df.drop('segmentsCabinCode', axis=1)
    df = df.drop('startingAirport', axis=1)
    df = df.drop('destinationAirport', axis=1)
    df = df.drop('departure_hour', axis=1)
    order = ['travelDuration', 'isBasicEconomy', 'isRefundable', 'isNonStop', 'totalTravelDistance', 'stopnumber', 'iscoach', 'ispremiumcoach', 'isfirst', 'isbusiness', 'departuremonth', 'departureday','departurehour','departureminute','arrivalmonth','arrivalday','arrivalhour','arrivalminute','startairportlabel','destinationairportlabel', 'airlinecodelabel']
    df = df[order]
    return df

model = joblib.load('xgb.pkl')
from sklearn.base import BaseEstimator, TransformerMixin
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return preprocessor(X)

pipeline = Pipeline([('transform_data', CustomPreprocessor()), ('model',model)])

joblib.dump(pipeline, 'xgb_pipeline.pkl')
