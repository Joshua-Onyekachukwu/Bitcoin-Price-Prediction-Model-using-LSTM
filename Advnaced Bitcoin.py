import os
import pandas as pd
import numpy as np
import requests
import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# --- Step 4: Split Data into Train and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Step 5: Build the LSTM Model ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# --- Step 6: Make Predictions ---
predicted_price_train = model.predict(X_train)
predicted_price_test = model.predict(X_test)

# Invert the scaling to get the actual price values
predicted_price_train = scaler.inverse_transform(predicted_price_train)
predicted_price_test = scaler.inverse_transform(predicted_price_test)

# --- Step 7: Model Evaluation ---

# Calculate the RMSE and R² for both train and test sets
rmse_train = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), predicted_price_train))
rmse_test = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predicted_price_test))

r2_train = r2_score(scaler.inverse_transform(y_train.reshape(-1, 1)), predicted_price_train)
r2_test = r2_score(scaler.inverse_transform(y_test.reshape(-1, 1)), predicted_price_test)

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")
print(f"Train R²: {r2_train}")
print(f"Test R²: {r2_test}")

# --- Step 8: Visualize the Results with Plotly ---
# Create interactive chart for actual vs predicted prices (for both train and test sets)
fig = go.Figure()

# Actual train data
fig.add_trace(go.Scatter(x=maindf.index[time_step:len(y_train)+time_step], y=scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(), mode='lines', name='Actual Train Price', line=dict(color='blue')))

# Predicted train data
fig.add_trace(go.Scatter(x=maindf.index[time_step:len(y_train)+time_step], y=predicted_price_train.flatten(), mode='lines', name='Predicted Train Price', line=dict(color='red')))

# Actual test data
fig.add_trace(go.Scatter(x=maindf.index[len(y_train)+time_step:len(y_train)+time_step+len(y_test)], y=scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), mode='lines', name='Actual Test Price', line=dict(color='green')))

# Predicted test data
fig.add_trace(go.Scatter(x=maindf.index[len(y_train)+time_step:len(y_train)+time_step+len(y_test)], y=predicted_price_test.flatten(), mode='lines', name='Predicted Test Price', line=dict(color='orange')))

# Layout configuration
fig.update_layout(
    title="Bitcoin Price Prediction using LSTM",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_dark"
)

# Show the plot
fig.show()
