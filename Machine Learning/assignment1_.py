import numpy as np
import pandas as pd
import plotly.express as px

# ---- read csv ----
data = pd.read_csv("risk_factors_cervical_cancer.csv")
data.replace('?',np.NaN,inplace=True)

# ---- Features and Target ----
X = data.select_dtypes(exclude=['int64']) # features; only continuous variables
y = data["Biopsy"] # target

# ---- Data Partitioning ----
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, shuffle = True)

# ---- Data Pre-Processing ----
 
# Impute with median
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(X_train)
X_train = pd.DataFrame(imp.transform(X_train), columns=X_train.columns)
imp.fit(X_test)
X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

# pre-process
from sklearn import preprocessing

cat_var = ["Smokes", "Hormonal Contraceptives", "IUD", "STDs", "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV"]

enc = preprocessing.OneHotEncoder(categories='auto',handle_unknown='ignore')

# Fit the encoder on the training data and transform the training set
df_hd_named_enc_train = pd.DataFrame(enc.fit_transform(X_train[cat_var]).toarray())
new_cols = enc.get_feature_names_out(cat_var)
df_hd_named_enc_train.columns = new_cols


X_train = X_train.drop(columns=cat_var)
X_train = pd.concat([X_train, df_hd_named_enc_train], axis=1)


df_hd_named_enc_test = pd.DataFrame(enc.transform(X_test[cat_var]).toarray())
df_hd_named_enc_test.columns = new_cols

X_test = X_test.drop(columns=cat_var) 
X_test = pd.concat([X_test, df_hd_named_enc_test], axis=1)

# Scale
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train),columns = X_train.columns)
X_test = pd.DataFrame(ss.fit_transform(X_test),columns = X_test.columns)

# ---- Hyperparameter Tuning ----
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}]
gs = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='f1')
gs.fit(X_train, y_train)
f = gs.best_params_['kernel']

# ---- Final Model Training and Evaluation ----

# Fit model and predict
model = SVC(class_weight='balanced', kernel=f)

model.fit(X_train, y_train)

y_predict_test = model.predict(X_test)


# Evaluate model
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, balanced_accuracy_score

f1 = f1_score(y_test, y_predict_test)


precision = precision_score(y_test, y_predict_test)
recall = recall_score(y_test, y_predict_test) 

fpr, tpr, thresholds = roc_curve(y_test, y_predict_test)
roc_auc = auc(fpr, tpr)

balanced_accuracy = balanced_accuracy_score(y_test, y_predict_test)

# Print statements
print(f"The chosen kernel for the final model is: {f}")
print(f"F1 Score: {f1:.4f}") 
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

