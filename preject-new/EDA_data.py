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


import random
def set_seed (seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
set_seed(42)





# setting pandas :

# تمام ستون‌ها رو نشون بده (بدون ... وسط)
pd.set_option('display.max_columns', None)

# تمام ردیف‌ها رو نشون بده
pd.set_option('display.max_rows', None)

# عرض هر ستون رو کامل نشون بده (بریده نشه)
pd.set_option('display.max_colwidth', None)



uf = pd.read_csv("bank-additional-full.csv" ,sep=';')



# print(uf['y'].value_counts(normalize=True)*100)

uf = pd.get_dummies(uf , drop_first=True)

uf['y_yes'] = uf['y_yes'].astype(int)

print(uf['y_yes'].value_counts())

# print(uf.corr()['y_yes'].abs().sort_values(ascending=False))

# print(uf.var().sort_values(ascending=False))



var_cols = uf.var()

drop_cols_var = var_cols[var_cols > 0.000073 ].index.tolist()

uf_clean = uf[drop_cols_var]

print(uf_clean.var().sort_values(ascending=False))

print(uf_clean.corrwith(uf_clean['y_yes']).abs().sort_values(ascending=False))

corr_cols = uf_clean.corr()['y_yes'].abs().sort_values(ascending=False)

drop_corr_cols = corr_cols[corr_cols >= 0.001003 ].index.tolist()

uf_clean = uf_clean[drop_corr_cols]

print(uf_clean.corrwith(uf_clean['y_yes']).abs().sort_values(ascending=False))

print(uf_clean['y_yes'].value_counts())

for i in uf_clean.columns :
    sns.boxplot(x=uf_clean[i] , data=uf_clean)
    plt.show()

