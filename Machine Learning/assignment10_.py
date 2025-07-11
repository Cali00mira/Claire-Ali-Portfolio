import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
import shap

#load data
data = pd.read_csv('risk_factors_cervical_cancer.csv')

continous_vars = ['Age','Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
                  'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)',
                  'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis', 
                  'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 
                  'STDs:syphilis', 'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum',
                  'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV']

target = ['Biopsy']

X = data[continous_vars]
X = X.replace('?', float('NaN'))
X = X
y = data[target].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11,shuffle=True)

# Impute categorical columns with the most frequent value
imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
categorical = ["STDs", "STDs:condylomatosis", "STDs:cervical condylomatosis",
               "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis",
               "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes",
               "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:HPV", 
               "STDs:Hepatitis B", "IUD", "Hormonal Contraceptives", "Smokes"]

imp_cat.fit(x_train[categorical])
x_train_cat = pd.DataFrame(imp_cat.transform(x_train[categorical]), columns=categorical).astype("category")
x_test_cat = pd.DataFrame(imp_cat.transform(x_test[categorical]), columns=categorical).astype("category")

# impute numerical columns with median values
imp_num = SimpleImputer(missing_values=np.nan, strategy='median')
numerical = ["Age", "Number of sexual partners", "First sexual intercourse",
                  "Num of pregnancies", "Smokes (years)", "Smokes (packs/year)", 
                  "Hormonal Contraceptives (years)", "IUD (years)", "STDs (number)"]
 
imp_num.fit(x_train[numerical])
x_train_num = pd.DataFrame(imp_num.transform(x_train[numerical]), columns=numerical)
x_test_num = pd.DataFrame(imp_num.transform(x_test[numerical]), columns=numerical)

X_train = pd.concat([x_train_cat, x_train_num], axis=1).reindex(x_train_cat.index)
X_test = pd.concat([x_test_cat, x_test_num], axis=1).reindex(x_test_cat.index)

# oversample minority class
categorical_indices = [X_train.columns.get_loc(col) for col in categorical]
oversample = SMOTENC(categorical_features=categorical_indices, sampling_strategy=0.7)

# Apply SMOTENC to the training data
X_train_r1, y_train_r1 = oversample.fit_resample(X_train, y_train)

label_encoder = LabelEncoder()

for col in categorical:
    X_train_r1[col] = label_encoder.fit_transform(X_train_r1[col].astype(str))
    X_test[col] = label_encoder.fit_transform(X_test[col].astype(str))

X_train_r1 = X_train_r1.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# fit model
model = XGBClassifier()
model.fit(X_train_r1, y_train_r1)

y_predict_test = model.predict(X_test)

# evaluation
f1 = f1_score(y_test, y_predict_test)
precision = precision_score(y_test, y_predict_test)
recall = recall_score(y_test, y_predict_test) 
fpr, tpr, thresholds = roc_curve(y_test, y_predict_test)
roc_auc = auc(fpr, tpr)
accuracy = accuracy_score(y_test, y_predict_test)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}") 
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC: {roc_auc:.4f}")

# plots
shap.initjs()

explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)