import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)
train = pd.read_csv("credit_train.csv")
#test = pd.read_csv("credit_train.csv")


useless_col = ["ID","Customer_ID","Month","Name","SSN","Num_Credit_Card","Interest_Rate","Type_of_Loan",
               "Changed_Credit_Limit","Num_Credit_Inquiries","Credit_Mix",'Credit_Utilization_Ratio',
               'Payment_of_Min_Amount','Total_EMI_per_month']

train = train.drop(columns=useless_col)

Occupation_list = (train["Occupation"].value_counts())
train["Occupation"] = train["Occupation"].replace("_______","No Job")
train["Annual_Income"] = train["Annual_Income"].str.replace("_", "").astype(float).round().astype(int)
train["Monthly_Inhand_Salary"].fillna(0,inplace=True)
train["Monthly_Inhand_Salary"] = train["Monthly_Inhand_Salary"].round().astype(int)
train["Num_Bank_Accounts"] = train["Num_Bank_Accounts"].replace(-1,0)

train["Num_of_Loan"] = train["Num_of_Loan"].str.replace("_", "").astype(float).round().abs().astype(int)
train["Delay_from_due_date"] = train["Delay_from_due_date"].abs()
train["Num_of_Delayed_Payment"] = train["Delay_from_due_date"].abs().astype(str).replace("_","")
train["Outstanding_Debt"] = train["Outstanding_Debt"].astype(str).str.replace("_","").astype(float).round().astype(int)
train['Credit_History_Age'] = train['Credit_History_Age'].astype(str).str.replace(' Years and ','.')
train['Credit_History_Age'] = train['Credit_History_Age'].astype(str).str.replace('Months','')
train = train[train["Credit_History_Age"] != "nan"]
train['Credit_History_Age'] = train['Credit_History_Age'].astype(float).round().astype(int)


train["Amount_invested_monthly"] = train["Amount_invested_monthly"].astype(str).str.replace("__","")
train["Amount_invested_monthly"] = train["Amount_invested_monthly"].replace("nan", 0).astype(float).round().astype(int)
train["Payment_Behaviour"] = train["Payment_Behaviour"].replace("!@9#%8", "Low_spent_Small_value_payments")
train["Monthly_Balance"].fillna(0,inplace=True)
train["Monthly_Balance"] = train["Monthly_Balance"].replace("__-333333333333333333333333333__", 0).astype(float).round().astype(int)

train["Age"] = train["Age"].astype(str).str.replace("_","").astype(int)
train = train[(train['Age'] >= 0) & (train['Age'] <= 100)]


train_obj = train.select_dtypes(object)
train_numeric = train.select_dtypes(int).describe()


train["Payment_Behaviour"] = train_obj["Payment_Behaviour"].map({
    "High_spent_Large_value_payments": 0,
    "High_spent_Medium_value_payments": 1,
    "High_spent_Small_value_payments": 2,
    "Low_spent_Large_value_payments": 3,
    "Low_spent_Medium_value_payments": 4,
    "Low_spent_Small_value_payments": 5
})
y = train["Credit_Score"]
y=y.map({"Poor":0,"Standard":1,"Good":2})
train.drop(columns=["Credit_Score"],inplace=True)
encoder = OneHotEncoder(sparse=False)
encoded_features = pd.DataFrame(encoder.fit_transform(train_obj[["Occupation"]]), columns=encoder.get_feature_names_out(["Occupation"]))
train.reset_index(drop=True, inplace=True)
train = pd.concat([train, encoded_features], axis=1)
train.drop(columns=["Occupation"],inplace=True)
binary_cols = [col for col in train.columns if train[col].nunique() == 2]
SS = StandardScaler()
non_binary_cols = [col for col in train.columns if col not in binary_cols]

for column in non_binary_cols:
    train[column] = SS.fit_transform(train[[column]])



X = train


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initializing the classifier
knn = KNeighborsClassifier()
params_knn = {'n_neighbors': [3,5,7,9,11,13,15,17,19,21]}  # You can expand this list if needed

grid_search_knn = GridSearchCV(knn, params_knn, cv=5, scoring='accuracy', verbose=1,n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
print(f"Best parameters for kNN: {grid_search_knn.best_params_}")
print(f"KNN score: {grid_search_knn.score(X_test, y_test)}")
knn_best = grid_search_knn.best_estimator_


lr = LogisticRegression(max_iter=10000)  # Setting a high iteration limit to ensure convergence
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Regularization strength
grid_search_lr = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy', verbose=1,n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
print(f"Best parameters for LR: {grid_search_lr.best_params_}")
print(f"LR score{grid_search_lr.score(X_test, y_test)}")
lr_best = grid_search_lr.best_estimator_
rf = RandomForestClassifier()
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Random Forest score: {grid_search_rf.score(X_test, y_test)}")
rf_best = grid_search_rf.best_estimator_

# Training Gradient Boosting Machine
gbm = GradientBoostingClassifier()
gbm_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}
grid_search_gbm = GridSearchCV(gbm, gbm_params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_gbm.fit(X_train, y_train)
print(f"Best parameters for GBM: {grid_search_gbm.best_params_}")
print(f"GBM score: {grid_search_gbm.score(X_test, y_test)}")
gbm_best = grid_search_gbm.best_estimator_

def perform_feature_selection(model, X_train, y_train):
    sfs = SFS(model, 
              k_features=20,
              forward=False, 
              floating=False, 
              verbose=2,
              scoring='accuracy',
              cv=5,n_jobs=-1)
    sfs = sfs.fit(X_train, y_train)
    
    return sfs
sfs_knn = perform_feature_selection(knn_best, X_train, y_train)
print('Selected features for kNN:', sfs_knn.k_feature_names_)

# Feature selection for LR
sfs_lr = perform_feature_selection(lr_best, X_train, y_train)
print('Selected features for LR:', sfs_lr.k_feature_names_)

def evaluate_model(model, X_train, X_test, y_train, y_test, features):
    model.fit(X_train[features], y_train)
    preds = model.predict(X_test[features])
    accuracy = accuracy_score(y_test, preds)
    return accuracy

# Evaluate all three models with selected features
knn_accuracy = evaluate_model(knn_best, X_train, X_test, y_train, y_test, list(sfs_knn.k_feature_names_))
print(f"Accuracy of kNN using selected features: {knn_accuracy:.2f}")

lr_accuracy = evaluate_model(lr_best, X_train, X_test, y_train, y_test, list(sfs_lr.k_feature_names_))
print(f"Accuracy of Logistic Regression using selected features: {lr_accuracy:.2f}")

# Re-tune hyperparameters using selected features
grid_search_knn = GridSearchCV(knn, params_knn, cv=5, scoring='accuracy', verbose=1,n_jobs=-1)
grid_search_knn.fit(X_train[list(sfs_knn.k_feature_names_)], y_train)
print(f"Retuned Best parameters for kNN: {grid_search_knn.best_params_}")
print(f"Retuned KNN score: {grid_search_knn.score(X_test[list(sfs_knn.k_feature_names_)], y_test)}")

grid_search_lr = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy', verbose=1,n_jobs=-1)
grid_search_lr.fit(X_train[list(sfs_lr.k_feature_names_)], y_train)
print(f"Retuned Best parameters for LR: {grid_search_lr.best_params_}")
print(f"Retuned LR score: {grid_search_lr.score(X_test[list(sfs_lr.k_feature_names_)], y_test)}")

grid_search_gbm = GridSearchCV(gbm, gbm_params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_gbm.fit(X_train, y_train) # Assuming you want to fit on the entire feature set.
print(f"Best parameters for GBM: {grid_search_gbm.best_params_}")
print(f"GBM score: {grid_search_gbm.score(X_test, y_test)}")

# You can then fit and print results for the Random Forest in a similar fashion:
grid_search_rf = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Random Forest score: {grid_search_rf.score(X_test, y_test)}")