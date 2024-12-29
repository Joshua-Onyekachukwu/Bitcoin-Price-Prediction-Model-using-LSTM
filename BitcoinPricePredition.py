import os
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

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
# Assuming the file is saved with the name 'bitcoin_price_data_YYYYMMDD_HHMMSS.csv'
maindf = pd.read_csv(filename)

# Convert the 'date' column to datetime
maindf['date'] = pd.to_datetime(maindf['date'])

# Check for null values
print('Null Values:', maindf.isnull().values.sum())
print('NA values:', maindf.isnull().values.any())

# --- Step 3: Prepare Data for LSTM ---
# Drop any missing values
maindf = maindf.dropna()

# Set 'date' as the index
maindf.set_index('date', inplace=True)

# Use the 'price' column as the target variable for prediction
data = maindf[['price']].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# Prepare the data for the LSTM model
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


time_step = 60  # Use the past 60 days to predict the next day's price
X, y = create_dataset(scaled_data, time_step)

# Reshape the data to be compatible with LSTM input (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --- Step 4: Build the LSTM Model ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# --- Step 5: Make Predictions ---
# Predict the prices using the model
predicted_price = model.predict(X)

# Invert the scaling to get the actual price values
predicted_price = scaler.inverse_transform(predicted_price)

# --- Step 6: Visualize the Results ---
# Plot the actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(maindf.index[time_step:], data[time_step:], color='blue', label='Actual Price')
plt.plot(maindf.index[time_step:], predicted_price, color='red', label='Predicted Price')
plt.title('Bitcoin Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

