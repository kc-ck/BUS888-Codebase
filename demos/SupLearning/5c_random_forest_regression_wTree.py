import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Set a random seed for reproducibility
np.random.seed(0)

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# y = 2 * x + 3 + noise
y = np.dot(X, np.array([2])) + 3 + np.random.normal(0, 1, len(X))
reg = RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y.ravel())

print(reg.score(X, y))
print(reg.feature_importances_)

# Predicting for a new data point
new_point = np.array([[9]])
prediction = reg.predict(new_point)
print(f'Prediction for {new_point[0]} is {prediction[0]}')

# Plotting the original data points
plt.scatter(X[:, 0], y, color='blue', label='Original data')

# Plotting the regression line
x_range = np.linspace(min(X[:, 0]), max(X[:, 0]), num=100).reshape(-1, 1)
y_range = reg.predict(x_range)
plt.plot(x_range, y_range, color='red', label='Random Forest fit')

# Plotting the predicted data point
plt.scatter(new_point[:, 0], prediction, color='green', label='Predicted data')

plt.title('Random Forest Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Extract one tree from the forest
tree = reg.estimators_[9]

fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (4,4), dpi=300)
plot_tree(tree,
          feature_names=['feature1'],
          filled=True)

plt.show()
