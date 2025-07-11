import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import plot_importance
import matplotlib.pylab as plt

# Features and Target
diabetes= load_diabetes()
X = diabetes['data']
y = diabetes['target']

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)

# Parameter tuning
model = xgb.XGBRegressor()
param_tuning = {
    'max_depth': np.arange(1,6,1),
    'learning_rate': np.arange(0.1,1,0.1),
    'n_estimators': np.arange(10,110,10),
    'reg_alpha': np.arange(0,1.1,0.1),
    'reg_lambda': np.arange(0,1.1,0.1)
}

random_search = RandomizedSearchCV(model, param_distributions=param_tuning, n_iter=100, cv=5, scoring="neg_mean_squared_error", random_state=0) 
random_search.fit(X_train, y_train)
best = random_search.best_params_ 
 
print(f"The best hyperparameters are: {best}")

# Final Model
final = xgb.XGBRegressor(**best)
final.fit(X_train,y_train)
y_pred = final.predict(X_test)

MSE = mean_squared_error(y_test,y_pred) 
MAE = mean_absolute_error(y_test,y_pred) 

print(f"The mean squared error of the final model is: ",{MSE})
print(f"The mean absolute error of the final model is: ",{MAE})

# Plot feature importance
feature_names = diabetes.feature_names
final.get_booster().feature_names = feature_names
plot_importance(final.get_booster())
plt.show()