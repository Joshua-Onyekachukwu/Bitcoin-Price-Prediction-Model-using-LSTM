import os
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import ta  # Technical Analysis library for feature engineering

# --- Step 1: Scrape Bitcoin Price Data and Save as CSV ---
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    'vs_currency': 'usd',
    'days': '365',  # Get data for the last 365 days
    'interval': 'daily'  # Daily data
}

# Send a request to the API
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()

    # Extract the prices and timestamps
    prices = data['prices']  # List of [timestamp, price] pairs

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])

    # Convert the timestamp to a readable date format
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date

    # Drop the 'timestamp' column as it's not needed anymore
    df = df.drop(columns=['timestamp'])

    # Save the data to a CSV file
    filename = f"bitcoin_price_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Bitcoin price data saved to {filename}")
else:
    print("Failed to retrieve data from CoinGecko API")

# --- Step 2: Load and Preprocess the Data ---
maindf = pd.read_csv(filename)

# Convert the 'date' column to datetime
maindf['date'] = pd.to_datetime(maindf['date'])

# --- Step 3: Feature Engineering ---
# Add Technical Indicators (e.g., Moving Averages, RSI, MACD)
maindf['ma_7'] = maindf['price'].rolling(window=7).mean()  # 7-day moving average
maindf['ma_30'] = maindf['price'].rolling(window=30).mean()  # 30-day moving average
maindf['rsi'] = ta.momentum.RSIIndicator(maindf['price']).rsi()  # Relative Strength Index
maindf['macd'] = ta.trend.macd(maindf['price'])  # MACD
maindf['macd_signal'] = ta.trend.macd_signal(maindf['price'])  # MACD Signal line

# Drop rows with NaN values created by moving averages
maindf = maindf.dropna()

# --- Step 4: Prepare Data for LSTM ---
# Set 'date' as the index
maindf.set_index('date', inplace=True)

# Use the 'price', 'ma_7', 'ma_30', 'rsi', 'macd', and 'macd_signal' as features
features = maindf[['price', 'ma_7', 'ma_30', 'rsi', 'macd', 'macd_signal']].values

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Prepare the data for LSTM model
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, :])  # Include all features
        y.append(data[i, 0])  # Predict the 'price' (target variable)
    return np.array(X), np.array(y)

time_step = 60  # Use the past 60 days to predict the next day's price
X, y = create_dataset(scaled_data, time_step)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Step 5: Build the LSTM Model ---
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout for regularization
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))  # Dropout for regularization
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# --- Step 6: Make Predictions ---
# Predict the prices using the trained model
predicted_price = model.predict(X)

# --- Fix: Inverse Transform the Predicted Prices ---
# Inverse transform only the 'price' column of predicted prices
predicted_price_scaled = predicted_price.reshape(-1, 1)  # Reshape to a 2D array (55, 1)

# Create a full array with 6 columns to match the shape the scaler expects
# Since only the 'price' is predicted, set the other columns (ma_7, ma_30, rsi, macd, macd_signal) to NaN or default values
predicted_price_full = np.zeros((predicted_price_scaled.shape[0], 6))

# Insert the predicted price values in the first column
predicted_price_full[:, 0] = predicted_price_scaled.flatten()

# Now inverse transform the full array
predicted_price_inverse = scaler.inverse_transform(predicted_price_full)[:, 0]  # Get only the 'price' column after inverse transform

# --- Step 7: Visualize the Results Using Plotly ---
# Plot the actual vs predicted prices interactively
fig = go.Figure()

# Add actual prices
fig.add_trace(go.Scatter(x=maindf.index[time_step:], y=maindf['price'][time_step:], mode='lines', name='Actual Price', line=dict(color='blue')))

# Add predicted prices
fig.add_trace(go.Scatter(x=maindf.index[time_step:], y=predicted_price_inverse, mode='lines', name='Predicted Price', line=dict(color='red')))

# Update layout
fig.update_layout(
    title="Bitcoin Price Prediction using LSTM",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend_title="Price",
    template="plotly_dark"
)

# Show the interactive plot
fig.show()
