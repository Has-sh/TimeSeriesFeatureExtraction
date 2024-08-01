import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LearningRateScheduler

# Load and preprocess extracted features
data = pd.read_csv('extracted_features.csv')

# Normalize features
scaler = MinMaxScaler() # Scale features between 0 and 1
scaled_features = scaler.fit_transform(data) # Fit and transform data

# Prepare data for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 50
X, y = create_sequences(scaled_features, time_steps)

# Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Define LSTM model
# model = Sequential([
#     LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dense(X_train.shape[2])
# ])

# model.compile(optimizer='adam', loss='mean_squared_error')

# def lr_scheduler(epoch, lr):
#     if epoch > 10:
#         lr = lr * 0.5
#     return lr

# callback = LearningRateScheduler(lr_scheduler)

# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])

# model.save('lstm_model.h5')

loaded_model = load_model('lstm_model.h5')

# Generate future forecasts
future_steps = 500
last_sequence = scaled_features[-time_steps:].reshape(1, time_steps, scaled_features.shape[1])
forecast = []

for _ in range(future_steps):
    pred = loaded_model.predict(last_sequence)[0]
    forecast.append(pred)
    last_sequence = np.concatenate([last_sequence[:, 1:, :], pred.reshape(1, 1, -1)], axis=1)

# Convert to DataFrame and scale back
forecast = np.array(forecast).reshape(-1, scaled_features.shape[1])
forecast_rescaled = scaler.inverse_transform(forecast)

# Prepare time indices for plotting
historical_time = np.arange(len(scaled_features))
forecast_time = np.arange(len(scaled_features), len(scaled_features) + future_steps)

# Plot each feature individually with a line graph
for i in range(scaled_features.shape[1]):
    plt.figure(figsize=(14, 7))

    # Plot historical data for the feature
    plt.plot(historical_time, scaler.inverse_transform(scaled_features)[:, i], label=f'Historical Data - Feature {i+1}')

    # Plot forecasted data for the feature
    plt.plot(forecast_time, forecast_rescaled[:, i], label=f'Forecasted Data - Feature {i+1}', linestyle='--')

    plt.title(f'Time Series Forecast for Feature {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.show()
