# Session 3: Classification with Decision Trees and Random Forest

Welcome to Session 3! This session covers supervised learning techniques for classification, focusing on decision trees (CART) and Random Forest methods. You'll learn to build, evaluate, and optimize classification models using real-world data.

## 📚 Session Overview

This session introduces you to:
- Classification fundamentals
- Decision Tree algorithms (CART)
- Random Forest ensemble methods
- Model evaluation and optimization
- Feature importance analysis
- ROC curves and AUC metrics

## 📖 Main Content

### 🌳 CART & Random Forest Classification Demo

**Python Notebook Version:** [`notebooks/CART & RandomForest Demo_TitanicData.ipynb`](notebooks/CART%20&%20RandomForest%20Demo_TitanicData.ipynb)
**Python Script Version:** [`demos/SupLearning/CART_RandomForest_Titanic.py`](demos/SupLearning/cart_randomforest_titanic_demo.py)

**What you'll learn:**
- Load and preprocess classification data
- Handle categorical variables and missing values
- Build decision tree classifiers using CART algorithm
- Evaluate model performance with multiple metrics
- Visualize decision trees and understand model interpretability
- Implement Random Forest for improved predictions
- Compare single tree vs. ensemble performance

**Key Topics Covered:**

#### 1. **Data Loading and Preprocessing**
- Load Titanic dataset from online source
- Handle categorical variables (sex, embarked)
- Remove unnecessary columns
- Fill missing values with appropriate strategies
- Visualize data distributions with pairplots

#### 2. **Decision Tree Classification (CART)**
- Build decision tree classifier with specified parameters
- Understand Gini impurity criterion
- Control tree complexity with max_depth parameter
- Fit model on training data
- Make predictions on test data

#### 3. **Model Evaluation**
- Calculate accuracy, precision, recall, and F1-score
- Understand the trade-offs between different metrics
- Evaluate model performance on unseen data

#### 4. **Model Visualization and Interpretation**
- Plot decision tree structure
- Understand feature splits and decision boundaries
- Interpret tree nodes and leaf predictions
- Analyze feature importance

#### 5. **Model Optimization**
- Loop through different max_depth parameters
- Compare performance metrics across tree depths
- Visualize performance curves
- Identify optimal model complexity

#### 6. **ROC Analysis**
- Generate ROC curves for different tree depths
- Calculate Area Under Curve (AUC) scores
- Compare model performance using ROC analysis
- Understand ROC interpretation

#### 7. **Random Forest Implementation**
- Build Random Forest classifier
- Understand ensemble method concepts
- Compare single tree vs. Random Forest performance
- Analyze improved prediction accuracy

#### 8. **Advanced Analysis**
- Visualize individual trees in Random Forest
- Understand ensemble diversity
- Compare Random Forest ROC performance
- Analyze feature importance across ensemble

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:

1. **Understand Classification Fundamentals**
   - Differentiate between regression and classification problems
   - Identify appropriate use cases for tree-based methods

2. **Implement Decision Trees**
   - Build CART classifiers with scikit-learn
   - Tune hyperparameters for optimal performance
   - Interpret tree structure and decisions

3. **Evaluate Classification Models**
   - Calculate and interpret accuracy, precision, recall, F1-score
   - Generate and analyze ROC curves
   - Compare models using multiple evaluation metrics

4. **Apply Random Forest**
   - Understand ensemble learning concepts
   - Implement Random Forest classifiers
   - Compare single tree vs. ensemble performance

5. **Optimize Model Performance**
   - Use cross-validation techniques
   - Tune hyperparameters systematically
   - Identify best-performing model configurations

---

## 💡 Key Concepts

### Decision Trees (CART)
- **Classification and Regression Trees**
- Recursive binary splitting
- Gini impurity for split selection
- Tree pruning and complexity control

### Random Forest
- **Ensemble learning method**
- Bootstrap aggregating (bagging)
- Out-of-bag error estimation
- Feature randomness and diversity

### Model Evaluation
- **Confusion matrix analysis**
- Precision vs. Recall trade-offs
- ROC curves and AUC interpretation
- Cross-validation techniques

---

## 📊 Dataset

**Titanic Dataset:**
- **Source:** Historical passenger manifest
- **Target Variable:** Passenger survival (0/1)
- **Features:** Age, sex, class, fare, embarked port, etc.
- **Size:** ~800 training samples
- **Type:** Binary classification problem

---

## 🛠️ Prerequisites

- Completion of Session 1 (Python basics) and Session 2 (Regression)
- Understanding of basic statistics and data visualization
- Familiarity with pandas and matplotlib

---

## 📋 Instructions

1. **Setup:** Ensure you have all required packages installed:
   ```python
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```

2. **Follow the notebook:** Work through each cell sequentially
   - Read the markdown explanations
   - Execute code cells in order
   - Analyze the output and visualizations

3. **Experiment:** Try modifying parameters to see their effects:
   - Change `max_depth` values
   - Adjust `n_estimators` in Random Forest
   - Compare different evaluation metrics

4. **Reflect:** Consider the following questions:
   - Which model performs best and why?
   - How do you interpret the decision tree structure?
   - What are the trade-offs between interpretability and performance?

---

## 🚀 Next Steps

This session prepares you for:
- Advanced ensemble methods (Gradient Boosting, XGBoost)
- Feature selection and engineering
- Cross-validation and model selection
- Real-world classification applications in finance

---

**Happy Learning!**

*C. Kaligotla | BUS 888*
