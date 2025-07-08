# Session 2: Regression and Data Preprocessing with Scikit-Learn

Welcome to Session 2! This session covers essential data preprocessing techniques and various regression methods using Scikit-Learn. The notebooks and scripts are designed to be run in order, building your understanding step by step.

## 1. Data Preprocessing Basics

**File:** [`demos/scikit-learn-1-basicFunctions.py`](demos/scikit-learn-1-basicFunctions.py)

- Learn about standardization, normalization, and handling missing values.
- Practice splitting data into training and testing sets.
- Introduction to generating synthetic datasets.

---

## 2. Simple Linear Regression

**File:** [`notebooks/SimpleLinearReg.ipynb`](notebooks/SimpleLinearReg.ipynb)

- Fit a simple linear regression model to synthetic data.
- Visualize the regression line and make predictions for new data points.

---

## 3. Multiple Linear Regression

**File:** [`notebooks/MultipleLinearReg.ipynb`](notebooks/MultipleLinearReg.ipynb)

- Extend linear regression to multiple features.
- Visualize data and the regression plane in 3D.
- Predict outcomes for new data points.

---

## 4. Real-World Regression Example: Apple Stock

**File:** [`demos/scikit-learn-2-regressing-aapl.py`](demos/scikit-learn-2-regressing-aapl.py)  
**File:** [`notebooks/Scikit_reg_example.ipynb`](notebooks/Scikit_reg_example.ipynb)

- Fetch real stock data using Yahoo Finance.
- Apply linear regression to predict future stock prices.
- Compare predictions with actual values and visualize results.

---

## 5. Regularization: Ridge Regression

**File:** [`notebooks/SimpleRidgeRegression.ipynb`](notebooks/SimpleRidgeRegression.ipynb)

- Understand the concept of regularization to prevent overfitting.
- Apply Ridge regression and visualize its effect.

---

## 6. Regularization: Lasso Regression

**File:** [`notebooks/SimpleLasso.ipynb`](notebooks/SimpleLasso.ipynb)

- Learn about Lasso regression and its feature selection properties.
- Fit a Lasso model and interpret the results.

---

## 7. Model Selection: LassoCV

**File:** [`notebooks/LassoCV.ipynb`](notebooks/LassoCV.ipynb)

- Use cross-validation to automatically select the best regularization parameter for Lasso.
- Visualize and interpret the results.

---

## 8. Regression Analysis and Prediction with Statsmodels

**File:** [`notebooks/Appendix_RegressionModels.ipynb`](notebooks/Appendix_RegressionModels.ipynb)

- Comprehensive tutorial on linear regression and prediction using statsmodels OLS
- Learn to interpret detailed regression output including F-statistics, R-squared, and p-values
- Practice with real taxi fare data including data cleaning and outlier handling
- Model comparison and selection techniques
- Advanced model diagnostics: residual analysis, Q-Q plots, and fitted vs residuals plots
- Train-test split evaluation using MAE, RMSE, and MPE metrics
- Perfect complement to scikit-learn regression methods with focus on statistical interpretation

---

## Instructions

- Work through the files in the order above.
- Each notebook/script is self-contained and includes code, comments, and visualizations.
- The Appendix notebook (Section 8) provides deeper statistical insights and complements the scikit-learn approach.
- Make sure you have installed all required packages (`numpy`, `scikit-learn`, `matplotlib`, `yfinance`, `statsmodels`, `pandas`, `seaborn`).

Happy learning!