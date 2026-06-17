from dataclasses import dataclass  , asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report ,roc_auc_score, precision_recall_curve,recall_score,precision_score ,root_mean_squared_error, confusion_matrix , mean_squared_error , r2_score ,accuracy_score,log_loss,f1_score
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder , StandardScaler ,LabelEncoder
import optuna
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
import torch.optim as optim
import joblib
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


X = uf_clean.drop('y_yes',axis=1).to_numpy()
Y = uf_clean['y_yes'].to_numpy()

# print(X.shape)
# print(Y.shape)


work_X , test_X , work_Y , test_Y = train_test_split(X , Y , test_size=0.2 , random_state=42,stratify=Y)
train_X , val_X , train_Y , val_Y = train_test_split(work_X , work_Y , test_size=0.2 ,random_state=42,stratify=work_Y)


#--> 1 :
# uder_samp = RandomUnderSampler(random_state=42)
# train_X_under , train_Y_under = uder_samp.fit_resample(train_X , train_Y)


#---> 2 :
smote_nn = SMOTEENN(random_state=42)

# train_X_nn , train_Y_nn = smote_nn.fit_resample(train_X , train_Y)





base_line = RandomForestClassifier(n_estimators=230 , criterion='entropy' , max_depth=12 , min_samples_leaf=1 , min_samples_split=3,
                                   class_weight='balanced_subsample' , max_features='log2' , max_samples=0.6)


base_line.fit(train_X, train_Y)



y_proba_val = base_line.predict_proba(test_X)[: ,1]

thershold = np.linspace(0.01 , 0.99 , 100)
best_precision = 0
best_thershold = 0.5
best_recall = 0


for thr in thershold :
    y_prds = (y_proba_val >= thr).astype(int)
    recall = recall_score(test_Y , y_prds)
    precision = precision_score(test_Y , y_prds , zero_division=0)


    if recall >= 0.85 and precision > best_precision :
        best_precision = precision
        best_thershold = thr
        best_recall = recall




y_proba_test = base_line.predict_proba(test_X)[: ,1]
y_prd_test = (y_proba_test >= best_thershold).astype(int)





# prds = base_line.predict(test_X)
# proba = base_line.predict_proba(test_X)

print('recall :' , recall_score(test_Y , y_prd_test))
print('acc :' , accuracy_score(test_Y , y_prd_test))
print('precision :' , precision_score(test_Y , y_prd_test))
print('f1 :', f1_score(test_Y , y_prd_test))
print(confusion_matrix(test_Y , y_prd_test))
print(classification_report(test_Y , y_prd_test))
