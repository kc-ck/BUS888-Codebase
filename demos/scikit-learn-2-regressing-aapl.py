import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Parameters
verbose = True

# Fetch historical market data from Yahoo Finance
df = yf.download('AAPL', start='2023-05-01', end='2023-05-31', progress=False)
df_next = yf.download('AAPL', start='2023-06-01', end='2023-06-02', progress=False)

if verbose:
    output = df.head(5).to_string(index=False)
    print(output)

# Use only 'Close' and convert the dataframe to numpy array
X = np.arange(1, len(df) + 1).reshape(-1, 1)
y = np.array(df['Close'])

if verbose:
    print(X)
    print(y)

# Create a linear regression model and fit the data, get the slope and intercept
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_

# Predict the output for a new input based on the regression line
X_pred = np.array([[len(df)+1]])  # Reshape X_pred as a 2D array
y_pred = model.predict(X_pred)
y_pred_by_line = intercept + slope * X_pred

# Get the actual next data point
X_next = [[len(df)+1]]
y_next = df_next['Close'][0]

print("Prediction for", X_pred[0][0], "is:", y_pred[0],". Using a line, we get: ", y_pred_by_line)
print("Actual value for", X_next[0][0], "is:", y_next)

# Plot the data and regression line
X_range = np.arange(0, 35).reshape(-1, 1)
y_range = intercept + slope * X_range

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_range, y_range, color='red', linestyle='--', label='Regression Line')
plt.scatter(X_pred, y_pred, color='green', label='Prediction')
plt.scatter(X_next, y_next, color='orange', label='Actual')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
