from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load your data
df = pd.read_csv("../Datasets Clase 2/tb_sellout_01_todos.csv")

# Sort and filter data
df = df.sort_values(['product_id', 'periodo'])
df = df[df['periodo'] <= 201904]  # Assuming you have data up to 201904

# Create lagged features
df['lag_cust_request_qty'] = df.groupby('product_id')['cust_request_qty'].shift(2)
df['lag_cust_request_tn'] = df.groupby('product_id')['cust_request_tn'].shift(2)

# Drop NaNs created due to lagging
df = df.dropna()

# Separate into features and labels, and training and prediction sets
train_data = df[df['periodo'] < 201903]
predict_data = df[df['periodo'] == 201904]

X_train = train_data[['product_id', 'lag_cust_request_qty', 'lag_cust_request_tn']].values
y_train = train_data['tn'].values
X_predict = predict_data[['product_id', 'lag_cust_request_qty', 'lag_cust_request_tn']].values

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_predict = scaler.transform(X_predict)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_predict = np.reshape(X_predict, (X_predict.shape[0], 1, X_predict.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X_train, y_train, epochs=75, batch_size=32, verbose=1)

# Make predictions for 201904
predictions = model.predict(X_predict)

# Create a DataFrame for the final output
output = pd.DataFrame({'product_id': predict_data['product_id'].values, 'predicted_tn': predictions.flatten()})

# Merge with actual tn values for 201904 if available
# output = output.merge(df[df['periodo'] == 201904][['product_id', 'tn']], on='product_id', how='left')

# Save to CSV
output.to_csv('predictions_with_actual_lstm_lagged.csv', index=False)
