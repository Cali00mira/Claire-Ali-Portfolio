import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes


# Features and Target ----
diabetes= load_diabetes()
X = diabetes['data']
y = diabetes['target']

# Data Partitioning ----
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)

# one hot encoding
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder(categories='auto')

X_train_enc = enc.fit_transform(X_train[:,1].reshape(-1,1)).toarray()
X_train = np.delete(X_train,1,1)
X_train = np.concatenate((X_train_enc,X_train),axis=1)

X_test_enc = enc.transform(X_test[:,1].reshape(-1,1)).toarray()
X_test = np.delete(X_test,1,1)
X_test = np.concatenate((X_test_enc,X_test),axis=1)

# Hyperparameter tuning
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'kernel': ['linear', 'rbf', 'sigmoid','poly']},  
    {'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5]} 
]


gs = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
f = gs.best_params_
print(f"The chosen kernel for the final model is: {f}")

# fitting model
model = SVR(kernel=f['kernel'],degree=f['degree'])
model.fit(X_train, y_train)
y_predict_test = model.predict(X_test)

# Evaluate model
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_predict_test)
print(f"The Mean Squared Error is: {mse}")

# MAE
mae = mean_absolute_error(y_test, y_predict_test)
print(f"The Mean Absolute Error: {mae}")