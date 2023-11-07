from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

import pandas as pd



# Load your data
df = pd.read_csv("../Datasets Clase 2/tb_sellout_01_todos.csv")


# Create an empty DataFrame to store the final results

# Loop through each unique product_id to fit a separate LSTM model
rows = []
for product in df['product_id'].unique():
    print('Fitting model for product_id: {}'.format(product))
    # Filter data for the specific product
    product_data = df[df['product_id'] == product].sort_values('periodo')
    
    # Prepare data
    X = product_data[['cust_request_qty', 'cust_request_tn']].values
    y = product_data['tn'].values
    
    # Reshape data for LSTM
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Fit the model
    model.fit(X, y, epochs=50, batch_size=1, verbose=0)
    
    # Make prediction for period 201904 (assuming you have the features for it)
    cust_request_qty_201904 = product_data[product_data['periodo'] == 201904]['cust_request_qty'].values[0]
    cust_request_tn_201904 = product_data[product_data['periodo'] == 201904]['cust_request_tn'].values[0]
    X_predict = np.array([[cust_request_qty_201904, cust_request_tn_201904]])  # Replace these with actual values
    X_predict = np.reshape(X_predict, (X_predict.shape[0], 1, X_predict.shape[1]))
    
    forecast_lstm = model.predict(X_predict)
    
    # Append the prediction to the final DataFrame
    rows.append({'product_id': product, 'predicted_tn_lstm': forecast_lstm[0][0]})

# Merge the LSTM predictions with the actual values for 201904
final_lstm_predictions = pd.DataFrame(rows, columns=['product_id', 'predicted_tn_lstm', 'real_tn_lstm'])
final_lstm_predictions = final_lstm_predictions.merge(df[df['periodo'] == 201904][['product_id', 'tn']], on='product_id', how='left')
final_lstm_predictions.rename(columns={'tn': 'actual_tn'}, inplace=True)

# Save to CSV
final_lstm_predictions.to_csv('predictions_with_actual_lstm.csv', index=False)
