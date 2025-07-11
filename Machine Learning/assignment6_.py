import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Importing data
df = pd.read_csv("processed.cleveland.data", sep=',',header=None)
df.replace('?',np.NaN,inplace=True)
df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
X = df[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
Y = df["num"].to_frame()

# Imputing missing values
from sklearn.impute import SimpleImputer

imp_num = SimpleImputer(strategy='mean') 
imp_cat = SimpleImputer(strategy='most_frequent') 

X.loc[:, 'ca'] = imp_num.fit_transform(X[['ca']]).flatten() 
X.loc[:, 'thal'] = imp_cat.fit_transform(X[['thal']]).flatten() 

# One hot encoding
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical 

nominal_columns = ['sex','cp','fbs','restecg','exang','slope','thal']
enc = preprocessing.OneHotEncoder(categories='auto')
df_enc = pd.DataFrame(enc.fit_transform(
    X[nominal_columns]).toarray())
df_enc.columns = enc.get_feature_names_out(nominal_columns)
ohe_x = pd.concat([X, df_enc], axis=1)
ohe_x.drop(nominal_columns,axis=1, inplace=True)

Y_ohe = to_categorical(Y, num_classes = 5)
Y_ohe = pd.DataFrame(Y_ohe,columns=['0','1','2','3','4'])


# Scaling
from sklearn.preprocessing import StandardScaler

num_columns = ['age','trestbps','chol','thalach','oldpeak','ca']
ss = StandardScaler()
ohe_x[num_columns] = pd.DataFrame(ss.fit_transform(ohe_x[num_columns]))

# Data partitioning
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ohe_x, Y_ohe, test_size = 0.3, random_state = 0)

# Tuning hidden layer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

parameter_space = {
    'hidden_layer_sizes': [5,10,15,20,25,30,35,40,45,50]
}

mlp = MLPRegressor()
gs = GridSearchCV(mlp, parameter_space)
gs.fit(X_train, y_train)
best = gs.best_params_['hidden_layer_sizes']

# Training neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy

model = Sequential([Dense(best, activation='relu'),
                    Dense(5, activation='softmax') # 5 neurons in output
])

model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=32)
model.evaluate(X_test,  y_test)
