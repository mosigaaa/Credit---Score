# Project Overview
In this project, we utilize machine learning to predict the credit scores of bank customers based on their profile data. The chosen algorithm for this endeavor is an array of models, including K-Nearest Neighbors (KNN), Logistic Regression, Random Forest, and Gradient Boosting Machine


# Dataset


# Tools and Technology 🔧
- **Programming Language** :
Python 
- **Libraries** :
  -scikit-learn 
  -pandas 
  -matplotlib and seaborn (To visualize and interpret the dataset)

## Methodology ⚙️
### 1. Data Exploration and Visualization 📈
### 2. Data Preprocessing 🛠️
### 3. Model Development 🖥️
- Implement a series of models:
- K-Nearest Neighbors
- Logistic Regression
- Random Forest
- Gradient Boosting Machine
- Use an 80-20 split between training and test data.
### 4. Parameter Tuning 🔍
- Use GridSearchCV and other hyperparameter tuning methods to optimize each model's performance.
### 5. Model Evaluation 📝
- Apply classification metrics such as accuracy, precision, recall, and the F1 score.
Compare models to select the best one in terms of performance on the test data.
### 6. Feature Selection 🚀
- Deploy the Sequential Feature Selector (SFS) to enhance model accuracy.
Retrain the models using the chosen feature subsets for optimized results.
### 7. Conclusion 🎯

 # k-Nearest Neighbors (kNN)
    Best number of neighbors: 3
    Score: ~ 0.75

# Logistic Regression (LR)
    Best regularization strength (C parameter): 100
    Score: ~ 0.578
# Random Forest (RF)
    Best parameters: Maximum depth - None, Minimum samples split - 2, Number of estimators - 200
    Score: ~ 0.762
# Gradient Boosting Machines (GBM)
    Best parameters: Learning rate - 0.1, Maximum depth - 5, Number of estimators - 200
    Score: ~ 0.697
