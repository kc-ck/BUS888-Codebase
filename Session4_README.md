# Session 4: Advanced Regression Techniques - Lasso, Ridge, and Ensemble Methods

Welcome to Session 4! This session covers advanced regression techniques including regularization methods (Lasso and Ridge) and ensemble methods (Random Forest). You'll learn to handle overfitting, perform feature selection, and build robust regression models.

## 📚 Session Overview

This session introduces you to:
- Regularization fundamentals (Lasso and Ridge)
- Cross-validation for hyperparameter tuning
- Random Forest regression and visualization
- Model comparison and evaluation
- Feature importance analysis

## 📖 Main Content

### 🎯 Lasso Regression Demo

**Python Script:** [`demos/SupLearning/3_lasso.py`](demos/SupLearning/3_lasso.py)

- Implement Lasso regression with L1 penalty
- Visualize regression lines and predictions


### 🔍 Lasso with Cross-Validation

**Python Script:** [`demos/SupLearning/3b_lasso_crossvalidation.py`](demos/SupLearning/3b_lasso_crossvalidation.py)

- Use LassoCV for automatic alpha selection
- Implement k-fold cross-validation
- Compare performance across different alpha values
- Optimize model hyperparameters automatically


### 🏔️ Ridge Regression Demo

**Python Script:** [`demos/SupLearning/4_ridge_regression.py`](demos/SupLearning/4_ridge_regression.py)

- Implement Ridge regression with L2 penalty
- Handle multicollinearity issues
- Feature standardization importance


### 🌲 Random Forest Regression (1D)

**Python Script:** [`demos/SupLearning/5_random_forest_regression.py`](demos/SupLearning/5_random_forest_regression.py)

- Build Random Forest regression models
- Understand ensemble learning concepts
- Model robustness

### 🌲 Random Forest Regression (2D)

**Python Script:** [`demos/SupLearning/5b_random_forest_regression_2D.py`](demos/SupLearning/5b_random_forest_regression_2D.py)

- Extend Random Forest to multiple features
- Visualize 3D regression surfaces
- Handle multidimensional prediction problems


### 🌳 Random Forest with Tree Visualization

**Python Script:** [`demos/SupLearning/5c_random_forest_regression_wTree.py`](demos/SupLearning/5c_random_forest_regression_wTree.py)


- Visualize individual trees in Random Forest
- Understand tree structure and decision nodes
- Interpret model predictions at tree level



### 🔍 k-Nearest Neighbors Classification

**Python Script:** [`demos/SupLearning/6_knn_classification_iris.py`](demos/SupLearning/6_knn_classification_iris.py)

- Implement k-NN classification
- Understand distance-based learning
- Visualize decision boundaries
- Optimize k parameter selection


### 🏦 Comprehensive Classifier Comparison

**Python Script:** [`demos/SupLearning/7_loanapproval_SUP_CLassifiers.py`](demos/SupLearning/7_loanapproval_SUP_CLassifiers.py)

- Compare multiple classification algorithms
- Implement comprehensive model evaluation
- Use cross-validation for robust assessment
- Handle real-world dataset preprocessing

**Key Topics Covered:**
- Model comparison methodology
- Cross-validation techniques
- Performance metrics interpretation
- Hyperparameter tuning

---

## 💡 Key Concepts

### Regularization
- **Lasso (L1) Regression**: Automatic feature selection through sparsity
- **Ridge (L2) Regression**: Coefficient shrinkage for multicollinearity
- **Alpha parameter**: Controls regularization strength
- **Cross-validation**: Model selection and hyperparameter tuning

### Ensemble Methods
- **Random Forest**: Bootstrap aggregating for variance reduction
- **Feature randomness**: Improving model diversity
- **Out-of-bag estimation**: Built-in validation method
- **Feature importance**: Measuring predictor contribution

### Distance-Based Learning
- **k-Nearest Neighbors**: Instance-based learning
- **Decision boundaries**: Non-parametric classification
- **Hyperparameter selection**: Optimal k value selection

---

## 🛠️ Prerequisites

- Completion of Sessions 1-3
- Understanding of basic linear regression
- Familiarity with overfitting and model validation
- Knowledge of Python, numpy, and scikit-learn
- Basic understanding of classification concepts

---

## 📋 Instructions

1. **Setup:** Ensure you have all required packages installed:
   ```python
   pip install scikit-learn numpy matplotlib pandas seaborn
   ```

2. **Follow the scripts sequentially:**
   - Start with basic Lasso regression ([`3_lasso.py`](demos/SupLearning/3_lasso.py))
   - Progress to cross-validation ([`3b_lasso_crossvalidation.py`](demos/SupLearning/3b_lasso_crossvalidation.py))
   - Explore Ridge regression ([`4_ridge_regression.py`](demos/SupLearning/4_ridge_regression.py))
   - Build Random Forest models ([`5_random_forest_regression.py`](demos/SupLearning/5_random_forest_regression.py))
   - Experiment with 2D and tree visualization
   - Complete with classification examples

3. **Experiment with parameters:**
   - Try different alpha values in Lasso/Ridge
   - Adjust n_estimators in Random Forest
   - Modify cross-validation folds
   - Compare different random seeds

4. **Analyze results:**
   - Compare coefficient values across methods
   - Examine feature importance rankings
   - Evaluate prediction accuracy
   - Visualize model behavior

---

## 🔧 Code Highlights

### Lasso Regression
```python
from sklearn.linear_model import Lasso
reg = Lasso(alpha=0.1)
reg.fit(X, y)
print("Lasso coefficients:", reg.coef_)
```

### Cross-Validation
```python
from sklearn.linear_model import LassoCV
reg = LassoCV(cv=3, random_state=0)
reg.fit(X, y)
print("Best alpha:", reg.alpha_)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, random_state=0)
reg.fit(X, y)
print("Feature importance:", reg.feature_importances_)
```

### k-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
```

---


## 📚 Additional Resources

### Documentation
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Lasso and Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#lasso)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

### Further Reading
- "Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Pattern Recognition and Machine Learning" by Bishop
- "Hands-On Machine Learning" by Aurélien Géron

---

**Happy Learning!**

*C. Kaligotla | BUS 888*