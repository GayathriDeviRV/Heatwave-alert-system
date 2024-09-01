import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Streamlit page configuration
st.set_page_config(page_title="Heatwave Prediction App", layout="centered")

# Heading and Description
st.title("Heatwave Prediction App")
st.write("""
This application predicts maximum temperatures for the next 7 days using historical weather data and a GRU-based machine learning model.
It also checks if a heatwave is predicted based on the temperatures for the upcoming days.
""")

# Load the pre-trained model and scaler
model = load_model('heatwave_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to fetch historical weather data from WeatherAPI


def fetch_historical_weather_data(api_key, location, date):
    url = f"http://api.weatherapi.com/v1/history.json?key={
        api_key}&q={location}&dt={date}"
    response = requests.get(url)
    data = response.json()
    features = []
    for day in data['forecast']['forecastday']:
        features.append({
            'date': day['date'],
            'max_temp': day['day']['maxtemp_c'],
            'min_temp': day['day']['mintemp_c'],
            'avg_temp': day['day']['avgtemp_c'],
            'humidity': day['day']['avghumidity'],
            'heat_index': day['day'].get('heatindex_c', None),
            'uv': day['day']['uv'],
        })
    return pd.DataFrame(features)


# Fetch historical weather data
API_KEY = '65652209b11c49928c0132348243108'
LOCATION = 'India'
start_date = datetime(2023, 9, 1)
end_date = datetime.now()
date_list = pd.date_range(start=start_date, end=end_date)
weather_data = pd.DataFrame()
for date in date_list:
    date_str = date.strftime('%Y-%m-%d')
    try:
        daily_data = fetch_historical_weather_data(API_KEY, LOCATION, date_str)
        weather_data = pd.concat([weather_data, daily_data], ignore_index=True)
    except Exception as e:
        st.write(f"Failed to fetch data for {date_str}: {e}")

# Feature Selection
selected_features = ['max_temp', 'avg_temp', 'humidity', 'heat_index', 'uv']
data = weather_data[selected_features].fillna(0)
scaled_data = scaler.transform(data)


def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)


SEQ_LENGTH = 21
X = create_sequences(scaled_data, SEQ_LENGTH)

# Predicting the next 7 days using the final model
predictions = []
input_seq = scaled_data[-SEQ_LENGTH:]

for _ in range(7):
    input_seq_reshaped = np.reshape(
        input_seq, (1, SEQ_LENGTH, len(selected_features)))
    predicted_temp = model.predict(input_seq_reshaped)
    predictions.append(predicted_temp[0, 0])
    new_data_point = np.append(predicted_temp, input_seq[-1, 1:])
    input_seq = np.append(input_seq[1:], [new_data_point], axis=0)

predicted_temps = scaler.inverse_transform(
    [np.append([p], input_seq[-1, 1:]) for p in predictions])
predicted_temps = [temp[0] for temp in predicted_temps]

# Get the next 7 days' dates
last_date = pd.to_datetime(weather_data['date'].iloc[-1])
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=7, freq='D')

# Display predicted temperatures with dates
predicted_results = pd.DataFrame({'Date': future_dates.strftime(
    '%Y-%m-%d'), 'Predicted Max Temp (°C)': predicted_temps})
st.subheader("Predicted Temperatures for Next 7 Days")
st.dataframe(predicted_results)

# Visualization of Predicted Temperatures for Next 7 Days
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(predicted_results['Date'], predicted_results['Predicted Max Temp (°C)'],
        marker='o', linestyle='-', color='r')
ax.set_title('Predicted Maximum Temperature for the Next 7 Days')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.grid(True)
st.pyplot(fig)

# Check for heatwave alert
heatwave_threshold = 40
consecutive_days_required = 3

heatwave_alert = False
consecutive_days = 0
heatwave_dates = []

for i, temp in enumerate(predicted_temps):
    if temp > heatwave_threshold:
        consecutive_days += 1
        heatwave_dates.append(future_dates[i])
    else:
        consecutive_days = 0
        heatwave_dates = []

    if consecutive_days >= consecutive_days_required:
        heatwave_alert = True
        break

if heatwave_alert:
    st.subheader(f"Heatwave Alert!")
    st.write(f"High temperatures predicted for {
             consecutive_days_required} or more consecutive days:")
    st.write(f"Dates of heatwave: {heatwave_dates}")
else:
    st.subheader("No heatwave predicted in the next 7 days.")
