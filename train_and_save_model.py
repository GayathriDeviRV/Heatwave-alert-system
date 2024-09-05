import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import pickle
from tensorflow.keras.losses import MeanSquaredError

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
start_date = datetime.now().replace(year=datetime.now().year - 1)
end_date = datetime.now()
date_list = pd.date_range(start=start_date, end=end_date)
weather_data = pd.DataFrame()
for date in date_list:
    date_str = date.strftime('%Y-%m-%d')
    try:
        daily_data = fetch_historical_weather_data(API_KEY, LOCATION, date_str)
        weather_data = pd.concat([weather_data, daily_data], ignore_index=True)
    except Exception as e:
        print(f"Failed to fetch data for {date_str}: {e}")

# Feature Selection
selected_features = ['max_temp', 'avg_temp', 'humidity', 'heat_index', 'uv']
data = weather_data[selected_features].fillna(0)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences for training


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)


SEQ_LENGTH = 21
X, y = create_sequences(scaled_data, SEQ_LENGTH)
split = int(len(X) * 0.75)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the parameter grid for grid search
param_grid = {
    'n_units': [50, 150],
    'dropout_rate': [0.2, 0.3],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [8, 16],
    'epochs': [50, 100]
}

# Grid Search
results = []
for params in ParameterGrid(param_grid):
    model = Sequential()
    model.add(GRU(params['n_units'], return_sequences=True,
              input_shape=(SEQ_LENGTH, len(selected_features))))
    model.add(Dropout(params['dropout_rate']))
    model.add(GRU(params['n_units']))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1, kernel_regularizer=l2(0.01)))
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = np.mean(np.abs(y_test - y_pred.flatten()) <= 2.0) * 100
    r2 = r2_score(y_test, y_pred) if np.var(y_test) > 0 else float('nan')

    results.append({
        'params': params,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy
    })

best_result = min(results, key=lambda x: x['mse'])
best_params = best_result['params']

# Retrain the model with best parameters
model = Sequential()
model.add(GRU(best_params['n_units'], return_sequences=True,
          input_shape=(SEQ_LENGTH, len(selected_features))))
model.add(Dropout(best_params['dropout_rate']))
model.add(GRU(best_params['n_units']))
model.add(Dropout(best_params['dropout_rate']))
model.add(Dense(1, kernel_regularizer=l2(0.01)))
optimizer = Adam(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss=MeanSquaredError())

history = model.fit(
    X_train, y_train,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=0
)

# Save the model and scaler
model.save('heatwave_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
