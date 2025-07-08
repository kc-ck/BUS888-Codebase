"""
Session 3: Intro to ML | CART AND RANDOM FOREST
This demo shows the use of CART and Random Forest methods for Classification Prediction on TITANIC Data

C Kaligotla | BUS 888
"""

# Import only basic Python libraries first
import sys
import subprocess
import pkg_resources

def install_required_packages():
    """Install required packages if not already installed"""
    required_packages = [
        "scikit-learn",
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "graphviz"
    ]
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = []
    
    for package in required_packages:
        if package not in installed:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All packages installed successfully!")
    else:
        print("All required packages are already installed.")

def import_libraries():
    """Import all required libraries after ensuring they are installed"""
    global DecisionTreeClassifier, export_graphviz, plot_tree
    global RandomForestClassifier, load_iris, train_test_split
    global accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
    global pd, np, sns, graphviz, plt, warnings
    
    from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import graphviz
    import matplotlib.pyplot as plt
    import warnings
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    print("All libraries imported successfully!")

def load_and_explore_data():
    """Step 2 & 3: Load and explore the Titanic dataset"""
    print("=" * 60)
    print("STEP 2 & 3: LOADING AND EXPLORING DATA")
    print("=" * 60)
    
    # Load data from Dropbox
    titanic_df = pd.read_csv("https://www.dropbox.com/s/4tw0ttokdrl62qs/titanic_train.csv?dl=1")
    
    print("Dataset shape:", titanic_df.shape)
    print("\nColumns:", titanic_df.columns.tolist())
    print("\nDataset info:")
    print(titanic_df.info())
    print("\nDataset description:")
    print(titanic_df.describe())
    
    return titanic_df

def clean_data(titanic_df):
    """Step 4: Clean and prepare the data"""
    print("=" * 60)
    print("STEP 4: CLEANING AND PREPARING DATA")
    print("=" * 60)
    
    # Convert categorical variables to numerical variables
    titanic_df['sex'] = titanic_df['sex'].replace({'male': 0, 'female': 1})
    titanic_df['embarked'] = titanic_df['embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
    print("Converted categorical variables to numerical")
    
    # Remove unnecessary columns
    titanic_df = titanic_df.drop(['passengerId','name','ticket','cabin'], axis=1)
    print("Removed unnecessary columns")
    
    # Replace missing values with the mean of the column
    titanic_df.fillna(titanic_df.mean(), inplace=True)
    print("Filled missing values with column means")
    
    print("\nCleaned dataset info:")
    print(titanic_df.info())
    
    # Plot distributions
    print("\nCreating pairplot...")
    g = sns.pairplot(titanic_df, hue="survived")
    plt.show()
    
    return titanic_df

def split_data(titanic_df):
    """Step 5: Split data into train-test sets"""
    print("=" * 60)
    print("STEP 5: SPLITTING DATA INTO TRAIN-TEST SETS")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        titanic_df.drop('survived', axis=1),
        titanic_df['survived'],
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def build_decision_tree(X_train, X_test, y_train, y_test):
    """Steps 6-8: Build, fit, and analyze CART Decision Tree model"""
    print("=" * 60)
    print("STEPS 6-8: BUILDING AND ANALYZING DECISION TREE MODEL")
    print("=" * 60)
    
    # Create a decision tree classifier using the CART algorithm with max depth of 3
    dt = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    
    # Fit the model to the training data
    dt.fit(X_train, y_train)
    print("Decision tree model fitted")
    
    # Predict the labels for the testing data
    y_pred = dt.predict(X_test)
    
    # Calculate the prediction performance of the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Decision Tree Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-1 Score: {f1:.4f}")
    
    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=X_train.columns, class_names=['Not Survived', 'Survived'], filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()
    
    return dt

def optimize_decision_tree(X_train, X_test, y_train, y_test):
    """Step 9: Improve model performance by testing different depths"""
    print("=" * 60)
    print("STEP 9: OPTIMIZING DECISION TREE MODEL")
    print("=" * 60)
    
    # Define lists to store the results
    depths = list(range(1, 8))
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Loop over different tree depths
    for depth in depths:
        dt = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(depths, accuracy_scores, label='Accuracy', marker='o')
    plt.plot(depths, precision_scores, label='Precision', marker='s')
    plt.plot(depths, recall_scores, label='Recall', marker='^')
    plt.plot(depths, f1_scores, label='F1 score', marker='d')
    plt.xlabel('Tree depth')
    plt.ylabel('Metric score')
    plt.title('Decision Tree Performance vs Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot ROC curves for different depths
    print("\nPlotting ROC curves for different depths...")
    auc_scores = []
    fpr_list = []
    tpr_list = []
    
    for depth in depths:
        dt = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        y_probs = dt.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_scores.append(roc_auc)
    
    # Plot ROC curves for each depth
    plt.figure(figsize=(10, 8))
    for i, depth in enumerate(depths):
        plt.plot(fpr_list[i], tpr_list[i], label=f'Depth {depth} (AUC = {auc_scores[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Tree Depths')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

def build_random_forest(X_train, X_test, y_train, y_test):
    """Build and evaluate Random Forest model"""
    print("=" * 60)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    # Create a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Fit the model to the training data
    rf.fit(X_train, y_train)
    print("Random Forest model fitted")
    
    # Predict on the test set
    y_pred = rf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Random Forest Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Calculate ROC AUC
    y_probs = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return rf

def visualize_random_forest_trees(rf, X_train):
    """Visualize individual trees in the random forest"""
    print("=" * 60)
    print("VISUALIZING INDIVIDUAL TREES IN RANDOM FOREST")
    print("=" * 60)
    
    print(f"Plotting {len(rf.estimators_)} individual trees...")
    
    for i, tree in enumerate(rf.estimators_):
        plt.figure(figsize=(12, 8))
        plot_tree(tree, 
                 feature_names=X_train.columns, 
                 class_names=['Not Survived', 'Survived'], 
                 filled=True, 
                 rounded=True,
                 max_depth=3)  # Limit depth for readability
        plt.title(f"Decision Tree {i+1} in Random Forest")
        plt.show()

def main():
    """Main function to run the complete analysis"""
    print("CART AND RANDOM FOREST DEMO ON TITANIC DATA")
    print("=" * 60)
    
    # Step 1: Install packages if needed and import libraries
    try:
        install_required_packages()
        import_libraries()
    except Exception as e:
        print(f"Error: Could not install packages or import libraries: {e}")
        print("Please ensure all required packages are installed manually")
        return
    
    # Step 2: Load and explore data
    titanic_df = load_and_explore_data()
    
    # Step 3: Clean data
    titanic_df_clean = clean_data(titanic_df)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(titanic_df_clean)
    
    # Step 5: Build and analyze decision tree
    dt_model = build_decision_tree(X_train, X_test, y_train, y_test)
    
    # Step 6: Optimize decision tree
    optimize_decision_tree(X_train, X_test, y_train, y_test)
    
    # Step 7: Build and analyze random forest
    rf_model = build_random_forest(X_train, X_test, y_train, y_test)
    
    # Step 8: Visualize random forest trees
    visualize_random_forest_trees(rf_model, X_train)
    
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
