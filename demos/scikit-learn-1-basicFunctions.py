import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Example for standardization
scaler = preprocessing.StandardScaler()
X = np.random.rand(10,4)
X_scaled = scaler.fit_transform(X)
print(f'The scaled mean is: {X_scaled.mean(axis=0)}. The scaled variance is: {X_scaled.std(axis=0)}')

# Example for normalizaton
norm = preprocessing.Normalizer()
X_norm = norm.transform(X)
print(X_norm)

# Example for missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X_imputed = imputer.fit_transform([[3.14,np.nan],[3,4],[7,3]])
print([[3.14,np.nan],[3,4],[7,3]])
print(X_imputed)

# Splitting the data
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)  # Create a random dataset
print(X[:3])  # Print the first three rows of X
print(y[:3])  # Print the first three elements of y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # Split the data
print(f'X training set {X_train.shape}\nX testing set {X_test.shape}\ny training set {y_train.shape}\ny testing set {y_test.shape}')

#
# Linear regression
#
# Generate x-values from 1 to 20, and random y-values based on the x-values
np.random.seed(42)
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
X = np.arange(1, 21).reshape(-1, 1)
y = 2 + 3 * X + 10*np.random.randn(20, 1)

# Create a linear regression model and do a fit
model = LinearRegression()
model.fit(X, y)

# Predict the output for a new input
X_new = np.array([[30], [32]])
y_pred = model.predict(X_new)

print("Predictions for ", np.array_str(X_new).replace('\n', ''), " are: ", np.array_str(y_pred).replace('\n', ''))

# Get the slope and the intercept
slope = model.coef_[0][0]
intercept = model.intercept_[0]

# Plot the data and regression line
X_range = range(1, 35)
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_range, [slope * X + intercept for X in X_range], color='red', linestyle='--', label='Regression Line')
plt.scatter(X_new, y_pred, color='green', label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()