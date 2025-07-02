import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
import matplotlib.pyplot as plt

# Parameters
verbose = True
window_size = 10
training_range = 100
ticker_symbol = "SPY"
start_date = "2019-01-01"
end_date = "2021-06-14"

# Set Options
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

# Retrieve the historical prices from Yahoo Finance
data = yf.download(ticker_symbol, start=start_date, end=end_date)

offset = len(data) - window_size - training_range - 2

# Extract the "Close" prices from the retrieved data and convert them to a NumPy array
prices = np.array(data["Close"].tolist())

if verbose:
    print(prices)

# We trade as follows.If the prediction is greater than the current price we go long and otherwise short. At the
accountValue = 0
account_values = [accountValue]
correctPredictions = 0
totalPredictions = 0

for outer in range(0,offset):
    # Generate x_train and y_train sequences
    x_train = []
    y_train = []

    train_start = outer
    train_end = outer+ training_range

    for i in range(train_start, train_end):
        x_train.append(prices[i:i+window_size])
        y_train.append(prices[i+window_size])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict the next value using the sliding window
    last_window = prices[train_end:train_end+window_size]  # Last window from the prices data
    next_value = model.predict(last_window.reshape(1, -1))

    # Compare the predicted value with the actual value
    current_value = prices[train_end+window_size-1]
    actual_value = prices[train_end + window_size]

    # For trading, we have all the data here
    if current_value < next_value:
        accountValue += (actual_value - current_value)*100
    elif current_value > next_value:
        accountValue += (current_value - actual_value)*100

    if current_value < next_value and current_value < actual_value:
        correctPredictions += 1
    if current_value > next_value and current_value > actual_value:
        correctPredictions += 1

    totalPredictions += 1

    if verbose:
        print("Current price: ", current_value, ", Predicted next value: ", next_value)
        print("Actual next value:", actual_value)
        print("Account value: ", accountValue)

    # Append current account value to account value tracker
    account_values.append(accountValue)

print("The ending account value is: ", accountValue, ", Accuracy: ", correctPredictions/totalPredictions, ", Last price: ", actual_value)

# Plot the account value
plt.plot(account_values)
plt.xlabel('Time')
plt.ylabel('Account Value')
plt.title('Account Value Over Time')
plt.show()