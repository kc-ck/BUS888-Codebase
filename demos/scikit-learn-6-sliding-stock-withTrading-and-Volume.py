import numpy as np
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import matplotlib.pyplot as plt

# Parameters
verbose = True
window_size = 10
training_range = 100
ticker_symbol = "SPY"
start_date = "2023-01-01"
end_date = "2024-06-15"

def print2d(*args):
    formatted_args = []
    for arg in args:
        if isinstance(arg, float):
            formatted_arg = "{:.2f}".format(arg)
            formatted_args.append(formatted_arg)
        else:
            formatted_args.append(arg)
    print(*formatted_args)


# Set Options
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

# Retrieve the historical prices from Yahoo Finance
data = yf.download(ticker_symbol, start=start_date, end=end_date)

offset = len(data) - window_size - training_range - 2

# Extract the "Close" and "Volume" from the retrieved data and convert them to a NumPy array
prices = np.array(data["Close"].tolist())
volumes = np.array(data["Volume"].tolist())

if verbose:
    print(prices)

accountValue = 0.0
account_values = [accountValue]
correctPredictions = 0
totalPredictions = 0
totalLong = 0
totalShort = 0
totalTrades = 0
numberOfShares = 0

for outer in range(0, offset):
    # Generate x_train and y_train sequences
    x_train = []
    y_train = []

    train_start = outer
    train_end = outer+ training_range

    for i in range(train_start, train_end):
        x_train.append(np.concatenate((prices[i:i+window_size], volumes[i:i+window_size])))
        y_train.append(prices[i+window_size])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Train the model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Predict the next value using the sliding window
    last_window = np.concatenate((prices[train_end:train_end+window_size], volumes[train_end:train_end+window_size]))  # Last window from the prices data
    next_value = model.predict(last_window.reshape(1, -1))

    # Compare the predicted value with the actual value
    current_value = prices[train_end+window_size-1]
    actual_next_value = prices[train_end+window_size]

    if next_value > current_value:
        if actual_next_value > current_value:
            correctPredictions += 1
        if numberOfShares == 0:
            numberOfShares = 100
            totalLong += 1
            totalTrades += 1
            accountValue -= 100*current_value
        elif numberOfShares < 0:
            numberOfShares += 200
            totalLong += 1
            totalTrades += 1
            accountValue -= 200*current_value
    elif next_value < current_value:
        if actual_next_value < current_value:
            correctPredictions += 1
        if numberOfShares == 0:
            numberOfShares = -100
            totalShort += 1
            totalTrades += 1
            accountValue += 100 * current_value
        elif numberOfShares > 0:
            numberOfShares -= 200
            totalShort += 1
            totalTrades += 1
            accountValue += 200*current_value

    totalPredictions += 1

    print2d("Account value: ", accountValue + numberOfShares * current_value, ", Accuracy: ",
            (correctPredictions / totalPredictions), ", Long: ",
            (totalLong / totalTrades), ", Short: ", (totalShort / totalTrades))

    # Append current account value to account value tracker
    account_values.append(accountValue+ numberOfShares * current_value)

print("The ending account value is: ", accountValue+ numberOfShares * current_value, ", Accuracy: ", correctPredictions/totalPredictions, ", Last price: ", actual_next_value)

# Plot the account value
plt.plot(account_values)
plt.xlabel('Time')
plt.ylabel('Account Value')
plt.title('Account Value Over Time')
plt.show()
