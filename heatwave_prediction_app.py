import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
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

# Function to fetch historical weather data from WeatherAPI


def fetch_historical_weather_data(api_key, location, date):
    url = f"http://api.weatherapi.com/v1/history.json?key={
        api_key}&q={location}&dt={date}"
    response = requests.get(url)
    data = response.json()

    # Extracting necessary features
    features = []
    for day in data['forecast']['forecastday']:
        features.append({
            'date': day['date'],
            'max_temp': day['day']['maxtemp_c'],
            'min_temp': day['day']['mintemp_c'],
            'avg_temp': day['day']['avgtemp_c'],
            'humidity': day['day']['avghumidity'],
            # Use .get to avoid KeyError
            'heat_index': day['day'].get('heatindex_c', None),
            'uv': day['day']['uv'],
        })

    return pd.DataFrame(features)


# Fetching a large amount of historical weather data
API_KEY = '65652209b11c49928c0132348243108'  # Replace with your actual API key
LOCATION = 'India'
start_date = datetime(2023, 9, 1)  # Start date for historical data
end_date = datetime.now()  # End date for historical data

# Collecting data over multiple API calls
date_list = pd.date_range(start=start_date, end=end_date)
weather_data = pd.DataFrame()

# Loop through each date in the date range to fetch data
for date in date_list:
    date_str = date.strftime('%Y-%m-%d')
    try:
        daily_data = fetch_historical_weather_data(API_KEY, LOCATION, date_str)
        weather_data = pd.concat([weather_data, daily_data], ignore_index=True)
    except Exception as e:
        st.write(f"Failed to fetch data for {date_str}: {e}")

# Feature Selection
selected_features = ['max_temp', 'avg_temp', 'humidity',
                     'heat_index', 'uv']  # Selecting relevant features
# Handling missing values if heat_index is None
data = weather_data[selected_features].fillna(0)

# Data Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        # Predicting the max_temp as it's critical for heatwave
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)


# Adjust sequence length as needed (e.g., 21 days)
SEQ_LENGTH = 21  # Using last 21 days to predict the next day

# Create sequences for training
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Ensure enough data points for meaningful train-test split
if len(X) < 10:
    st.error(
        "Not enough data points for training and testing. Increase the number of days of data.")
    st.stop()

# Splitting data into training and test sets
split = int(len(X) * 0.75)  # Using 75% for training, 25% for testing
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the parameter grid for grid search (only GRU models)
param_grid = {
    'n_units': [50, 150],  # Number of GRU units
    'dropout_rate': [0.2, 0.3],  # Dropout rate to prevent overfitting
    'learning_rate': [0.001, 0.0005],  # Learning rate for Adam optimizer
    'batch_size': [8, 16],  # Batch size for training
    'epochs': [50, 100]  # Number of epochs
}

# Create a list to store results
results = []

# Grid Search over the parameter grid
for params in ParameterGrid(param_grid):

    # Build the model
    model = Sequential()
    model.add(GRU(params['n_units'], return_sequences=True,
              input_shape=(SEQ_LENGTH, len(selected_features))))
    # Add dropout to prevent overfitting
    model.add(Dropout(params['dropout_rate']))
    model.add(GRU(params['n_units']))
    # Add dropout to prevent overfitting
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1, kernel_regularizer=l2(0.01)))  # L2 regularization

    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')

    # Train the model
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0  # Suppress training output for clarity
    )

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = np.mean(np.abs(y_test - y_pred.flatten()) <=
                       2.0) * 100  # Using ±2°C for accuracy
    r2 = r2_score(y_test, y_pred) if np.var(y_test) > 0 else float('nan')

    results.append({
        'params': params,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy
    })

# Find the best parameters based on MSE
best_result = min(results, key=lambda x: x['mse'])

# Displaying best parameters and metrics
st.subheader("Model Evaluation Metrics")
st.write(f"Best parameters: {best_result['params']}")
st.write(f"Mean Squared Error (MSE): {best_result['mse']:.4f}")
st.write(f"Mean Absolute Error (MAE): {best_result['mae']:.4f}")
st.write(f"R-squared (R2): {best_result['r2']:.4f}")
st.write(f"Accuracy (within ±2°C): {best_result['accuracy']:.2f}%")

# Retrain the model using the best parameters on the entire training set
best_params = best_result['params']
model = Sequential()
model.add(GRU(best_params['n_units'], return_sequences=True,
          input_shape=(SEQ_LENGTH, len(selected_features))))
# Add dropout to prevent overfitting
model.add(Dropout(best_params['dropout_rate']))
model.add(GRU(best_params['n_units']))
# Add dropout to prevent overfitting
model.add(Dropout(best_params['dropout_rate']))
model.add(Dense(1, kernel_regularizer=l2(0.01)))  # L2 regularization

optimizer = Adam(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='mse')

# Train the best model
history = model.fit(
    X_train, y_train,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=0  # Suppress training output for clarity
)

# Final evaluation on the test set
y_pred_final = model.predict(X_test)

# Predicting the next 7 days using the final model
predictions = []
input_seq = scaled_data[-SEQ_LENGTH:]

for _ in range(7):
    input_seq_reshaped = np.reshape(
        input_seq, (1, SEQ_LENGTH, len(selected_features)))
    predicted_temp = model.predict(input_seq_reshaped)
    predictions.append(predicted_temp[0, 0])

    # Update the input sequence with the predicted temperature
    # Adding other features unchanged
    new_data_point = np.append(predicted_temp, input_seq[-1, 1:])
    input_seq = np.append(input_seq[1:], [new_data_point], axis=0)

# Inverse transform the predictions to the original scale
predicted_temps = scaler.inverse_transform(
    [np.append([p], input_seq[-1, 1:]) for p in predictions])

# Extract the temperature predictions
predicted_temps = [temp[0] for temp in predicted_temps]

# Get the next 7 days' dates
last_date = pd.to_datetime(weather_data['date'].iloc[-1])
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=7, freq='D')

# Display predicted temperatures with dates
predicted_results = pd.DataFrame(
    {'Date': future_dates.strftime('%Y-%m-%d'), 'Predicted Max Temp (°C)': predicted_temps})
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
heatwave_threshold = 40  # Define heatwave threshold
consecutive_days_required = 3  # Define number of consecutive days required

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
