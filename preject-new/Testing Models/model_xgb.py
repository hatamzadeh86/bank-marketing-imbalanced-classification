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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score




# 1. فرض می‌کنیم دیتات رو قبلاً تمیز کردی و توی uf_clean داری
# X = uf_clean.drop('y_yes', axis=1)
# Y = uf_clean['y_yes']

# 2. تقسیم داده (همون کاری که قبلاً کردی)
work_X, test_X, work_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
train_X, val_X, train_Y, val_Y = train_test_split(work_X, work_Y, test_size=0.2, random_state=42, stratify=work_Y)

# 3. ساخت دیتاست مخصوص LightGBM (برای سرعت بیشتر)
lgb_train = lgb.Dataset(train_X, train_Y)
lgb_val = lgb.Dataset(val_X, val_Y, reference=lgb_train)

# 4. تنظیم پارامترها (مقدارها رو می‌تونی تغییر بدی)
params = {
    'objective': 'binary',           # مسئله دو کلاسه
    'metric': 'auc',                 # معیار ارزیابی
    'boosting_type': 'gbdt',         # نوع بوستینگ
    'num_leaves': 31,                # تعداد برگ‌ها (می‌تونی کمتر/بیشتر کنی)
    'max_depth': -1,                 # -1 یعنی بدون محدودیت (امن‌تره با num_leaves)
    'learning_rate': 0.05,           # قدم‌های کوچیک‌تر برای دقت بیشتر
    'feature_fraction': 0.9,         # استفاده از ۹۰٪ ویژگی‌ها
    'bagging_fraction': 0.8,         # استفاده از ۸۰٪ داده‌ها
    'bagging_freq': 5,               # هر ۵ بار، یک بار نمونه‌برداری مجدد
    'verbose': 0,                    # برای اینکه خروجی اضافی نده
    'random_state': 42               # برای تکرارپذیری
}

# 5. آموزش مدل (با Early Stopping برای جلوگیری از Overfit)
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],   # دیتای Validation برای Early Stopping
    num_boost_round=200,               # حداکثر تعداد درخت (تکرار)
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]  # اگه ۵۰ تا دور متوالی بهبود نداشت، بس کن
)

# 6. ارزیابی روی داده تست
y_pred_prob = model.predict(test_X, num_iteration=model.best_iteration)  # بهترین مدل رو انتخاب کن
y_pred = (y_pred_prob > 0.5).astype(int)  # آستانه ۰.۵

print('recall   :', recall_score(test_Y, y_pred))
print('precision:', precision_score(test_Y, y_pred))
print('f1       :', f1_score(test_Y, y_pred))
print('accuracy :', accuracy_score(test_Y, y_pred))
print('auc      :', roc_auc_score(test_Y, y_pred_prob))
print('\nConfusion Matrix:\n', confusion_matrix(test_Y, y_pred))
print('\nClassification Report:\n', classification_report(test_Y, y_pred))


#_________________________________________________________________
# Training until validation scores don't improve for 50 rounds
# [50]	training's auc: 0.957115	valid_1's auc: 0.950749
# [100]	training's auc: 0.963821	valid_1's auc: 0.951652
# [150]	training's auc: 0.969735	valid_1's auc: 0.95177
# Early stopping, best iteration is:
# [132]	training's auc: 0.967818	valid_1's auc: 0.952011
# recall   : 0.5678879310344828
# precision: 0.6925098554533509
# f1       : 0.6240378922439314
# accuracy : 0.9229181840252488
# auc      : 0.955339284871928

# Confusion Matrix:
#  [[7076  234]
#  [ 401  527]]

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.95      0.97      0.96      7310
#            1       0.69      0.57      0.62       928

#     accuracy                           0.92      8238
#    macro avg       0.82      0.77      0.79      8238
# weighted avg       0.92      0.92      0.92      8238
