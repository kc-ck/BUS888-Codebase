import numpy as np
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(0)

# Create some input data
X = np.array([[1, 1], [2, 3], [3, 2], [4, 4], [5, 5], [6, 6], [7, 7], [8, 9]])
# y = 2*X1 + 3*X2 + 4 + noise
y = 2*X[:, 0] + 3*X[:, 1] + 4 + np.random.normal(0, 1, len(X))
reg = RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y.ravel())

print(reg.score(X, y))
print(reg.feature_importances_)

# Predicting for a new data point
new_point = np.array([[9, 10]])
prediction = reg.predict(new_point)
print(f'Prediction for {new_point[0]} is {prediction[0]}')

# Plotting the original data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Original data')

# Plotting the regression surface
x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), num=10)
x2_range = np.linspace(min(X[:, 1]), max(X[:, 1]), num=10)
x1_range, x2_range = np.meshgrid(x1_range, x2_range)

y_range = reg.predict(np.c_[x1_range.ravel(), x2_range.ravel()])
y_range = y_range.reshape(x1_range.shape)

ax.plot_surface(x1_range, x2_range, y_range, color='red', alpha=0.3)

# Plotting the predicted data point
ax.scatter(new_point[:, 0], new_point[:, 1], prediction, color='green', label='Predicted data')

plt.title('Random Forest Regression')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
