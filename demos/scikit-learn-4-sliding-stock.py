import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import yfinance as yf
import pandas as pd

# Some parameters
verbose = True
pd.set_option('display.max_columns', None)

# Fetch historical market data from yahoo finance
df = yf.download("AAPL", start='2023-06-11', end='2024-06-11', progress=False)



# Use only 'Close' and convert the dataframe to numpy array
dataset = df.filter(['Close']).values

# Scale the data - the scales requires a 2-dimensional array which we have here as a number of rows in one column
# Normally, each column represents a feature and the data can contain many features. Here, since we are just dealing
# with prices, we have just one feature. While this 'looks' similar to a one-dimensional array, there is a difference
# arr[0] accesses the first element in the one-dimensional array while arr[0,0] accesses the first element in a
# two-dimensional array
min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(dataset)

print(scaled_data[:10])
print(dataset.shape)
print(scaled_data.shape)

# Create the model
model = LinearRegression()

# Make predictions 30 days into the future using a sliding window approach
future_predictions = []
actual_values = []

for i in range(len(dataset) - 60, len(dataset) - 30):
    # Train the model - one needs to be a bit careful here ... since we are using machine learning, each past price
    # is like a feature so: the linear regression model expects the input to be a 2-dimensional array with shape
    # (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.
    # In this case, the model expects a single sample (row) with 30 features (columns) representing the 30
    # previous data points. he reshape(1, -1) operation is applied to convert the 2-dimensional array (30, 1)
    # into a single-row 2-dimensional array (1, 30). The -1 argument in reshape(1, -1) indicates that the number
    # of columns should be automatically inferred based on the size of the array and the specified row size
    # (which is 1 in this case).
    x_train = np.array([scaled_data[j] for j in range(i-30, i)]).reshape(1, -1)
    y_train = np.array([scaled_data[i]]).reshape(-1,)

    # print(x_train,y_train)

    model.fit(x_train, y_train)

    # Make prediction and add it to future_predictions
    x_test = np.array([scaled_data[j] for j in range(i-29, i+1)]).reshape(1, -1)
    prediction = model.predict(x_test)
    future_predictions.append(prediction[-1])

    # Add the actual value to actual_values
    actual_values.append(scaled_data[i+1][0])

# Reverse scaling
future_predictions = min_max_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
actual_values = min_max_scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))

# Plot the results
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(df.index[-30:], actual_values, marker='o')
plt.plot(df.index[-30:], future_predictions, marker='x')
plt.legend(['Actual', 'Prediction'], loc='lower right')
plt.show()
