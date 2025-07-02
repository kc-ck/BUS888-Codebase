import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

# Sample price data
prices = [10, 12, 15, 18, 20, 19, 22, 25, 28, 26, 30, 32]

# Convert prices to NumPy array
prices = np.array(prices)

# Sliding window size
window_size = 5

# Offset
offset = 3

# Generate x_train and y_train sequences
x_train = []
y_train = []

for i in range(len(prices) - window_size - offset):
    x_train.append(prices[i:i+window_size])
    y_train.append(prices[i+window_size])

x_train = np.array(x_train)
y_train = np.array(y_train)

print("x_train is the following 2D array:")
print(x_train)
print("y_train is the following 1D array:")
print(y_train)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Print the coefficients
print("We get the following: ")
print("Coefficients: ", model.coef_)
print("Intercept: {:.2f}".format(model.intercept_))

# Predict the next value using the sliding window
last_window = prices[-window_size-offset:-offset]  # Last window from the prices data
next_value = model.predict(last_window.reshape(1, -1))

print("The last window (used to predict a next value) is:")
print(last_window)

# Compare the predicted value with the actual value
actual_value = prices[-offset]

print("Prices used for the prediction (window): ", last_window)
print("Predicted next value: ", next_value)
print("Actual next value:", actual_value)


# Plot the prices along with the predicted and actual point
plt.plot(prices[:-offset], label='Prices')
plt.scatter(len(prices) - offset, actual_value, color='red', label='Actual')
plt.scatter(len(prices) - offset, next_value, color='green', label='Predicted')

# Adding labels and legend
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Displaying the plot
plt.show()