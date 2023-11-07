# Let's start by adapting the example code to your specific requirements.
# We'll start with Option 1: running a separate model for each `product_id`.

from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from pprint import pprint
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)
import os
pprint(os.environ)
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)
print("*"*100)


# Function to split a single time series into overlapping sequences
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# Read the data
df = pd.read_csv("../Datasets Clase 2/tb_sellout_01_todos.csv")

# Filter data up to 201902
df["lag_cust_request_qty"] = df.groupby("product_id")["cust_request_qty"].shift(2)
df["lag_cust_request_tn"] = df.groupby("product_id")["cust_request_tn"].shift(2)
df["lag_tn"] = df.groupby("product_id")["tn"].shift(2)
df = df.dropna()
df = df[df["periodo"] <= 201904]

# Initialize an empty DataFrame for the final output

# Number of time steps to use for each sequence
n_steps = 5
rows = []
# Loop through each unique `product_id` to train a separate model
for i,product in enumerate(df["product_id"].unique()):
    print(f"Training model for product {product} ({i+1}/{len(df['product_id'].unique())}))")
    # Filter data for the current `product_id`
    product_data = df[df["product_id"] == product]

    # Sort by `periodo` just to be sure
    product_data = product_data.sort_values("periodo")

    # Drop the columns that won't be used as features
    product_data = product_data[
        ["lag_cust_request_qty", "lag_cust_request_tn", "lag_tn", "tn"]
    ]

    # Convert DataFrame to NumPy array
    product_data_array = product_data.values

    # Prepare the sequences
    X, y = split_sequences(product_data_array, n_steps)

    # Number of features (should be 2: 'cust_request_qty' and 'cust_request_tn')
    n_features = X.shape[2]

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Fit the model
    model.fit(X, y, epochs=200, verbose=0)

    # Prepare the input for prediction
    x_input = product_data_array[-n_steps:, :-1]
    x_input = x_input.reshape((1, n_steps, n_features))

    # Make prediction
    yhat = model.predict(x_input, verbose=0)

    # Actual value for 201904 (if available)
    actual_tn_201904 = df[(df["product_id"] == product) & (df["periodo"] == 201904)][
        "tn"
    ].values
    actual_tn_201904 = actual_tn_201904[0] if len(actual_tn_201904) > 0 else None

    # Append to final output DataFrame
    rows.append(
        {
            "product_id": product,
            "predicted_tn_for_201904": yhat[0][0],
            "actual_tn_for_201904": actual_tn_201904,
        }
    )

# Display a sample of the final output
final_output = pd.DataFrame(rows,
    columns=["product_id", "predicted_tn_for_201904", "actual_tn_for_201904"]
)
final_output.to_csv("output_lstm4.csv", index=False)
final_output.head()


# Read the data
df = pd.read_csv("../Datasets Clase 2/tb_sellout_01_todos.csv")

# Filter data up to 201902
df["lag_cust_request_qty"] = df.groupby("product_id")["cust_request_qty"].shift(2)
df["lag_cust_request_tn"] = df.groupby("product_id")["cust_request_tn"].shift(2)
df["lag_tn"] = df.groupby("product_id")["tn"].shift(2)
df = df.dropna()
df = df[df["periodo"] <= 201904]

# Initialize an empty DataFrame for the final output

# Number of time steps to use for each sequence
n_steps = 5
rows = []
# Loop through each unique `product_id` to train a separate model
for i,product in enumerate(df["product_id"].unique()):
    print(f"Training model for product {product} ({i+1}/{len(df['product_id'].unique())}))")
    # Filter data for the current `product_id`
    product_data = df[df["product_id"] == product]

    # Sort by `periodo` just to be sure
    product_data = product_data.sort_values("periodo")

    # Drop the columns that won't be used as features
    product_data = product_data[
        ["lag_cust_request_qty", "lag_cust_request_tn", "lag_tn", "tn"]
    ]

    # Convert DataFrame to NumPy array
    product_data_array = product_data.values

    # Prepare the sequences
    X, y = split_sequences(product_data_array, n_steps)

    # Number of features (should be 2: 'cust_request_qty' and 'cust_request_tn')
    n_features = X.shape[2]

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Fit the model
    model.fit(X, y, epochs=200, verbose=0)

    # Prepare the input for prediction
    x_input = product_data_array[-n_steps:, :-1]
    x_input = x_input.reshape((1, n_steps, n_features))

    # Make prediction
    yhat = model.predict(x_input, verbose=0)

    # Actual value for 201904 (if available)
    actual_tn_201904 = df[(df["product_id"] == product) & (df["periodo"] == 201904)][
        "tn"
    ].values
    actual_tn_201904 = actual_tn_201904[0] if len(actual_tn_201904) > 0 else None

    # Append to final output DataFrame
    rows.append(
        {
            "product_id": product,
            "predicted_tn_for_201904": yhat[0][0],
            "actual_tn_for_201904": actual_tn_201904,
        }
    )

# Display a sample of the final output
final_output = pd.DataFrame(rows,
    columns=["product_id", "predicted_tn_for_201904", "actual_tn_for_201904"]
)
final_output.to_csv("output_lstm4.csv", index=False)
final_output.head()
