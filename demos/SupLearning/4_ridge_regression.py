import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set a random seed for reproducibility
np.random.seed(0)

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# y = 2 * x + 3 + noise
y = np.dot(X, np.array([2])) + 3 + np.random.normal(0, 1, len(X))
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
ridge = Ridge(alpha=1.0)  # Alpha is the Ridge penalty term
ridge.fit(X, y)

print(ridge.score(X, y))
print(ridge.coef_)
print(ridge.intercept_)

# Predicting for a new data point
new_point = scaler.transform(np.array([[9]]))  # Make sure to also standardize any new data
prediction = ridge.predict(new_point)
print(f'Prediction for {new_point[0]} is {prediction[0]}')

# Plotting the original data points
plt.scatter(X[:, 0], y, color='blue', label='Original data')

# Plotting the regression line
x_range = np.linspace(min(X[:, 0]), max(X[:, 0]), num=100)
y_range = ridge.coef_[0]*x_range + ridge.intercept_
plt.plot(x_range, y_range, color='red', label='Ridge regression line')

# Plotting the predicted data point
plt.scatter(new_point[:, 0], prediction, color='green', label='Predicted data')

plt.title('Ridge Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
