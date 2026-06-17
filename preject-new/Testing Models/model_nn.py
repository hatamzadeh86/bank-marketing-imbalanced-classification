from torchvision.ops import sigmoid_focal_loss

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



X = uf_clean.drop('y_yes',axis=1).to_numpy()
Y = uf_clean['y_yes'].to_numpy()

# print(X.shape)
# print(Y.shape)


work_X , test_X , work_Y , test_Y = train_test_split(X , Y , test_size=0.2 , random_state=42,stratify=Y)
train_X , val_X , train_Y , val_Y = train_test_split(work_X , work_Y , test_size=0.2 ,random_state=42,stratify=work_Y)


scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)
test_X = scaler.fit_transform(test_X)
val_X = scaler.fit_transform(val_X)



train_X = torch.tensor(train_X ,dtype=torch.float32)
train_Y = torch.tensor(train_Y ,dtype=torch.long)
test_X = torch.tensor(test_X ,dtype=torch.float32)
test_Y = torch.tensor(test_Y ,dtype=torch.long)
val_X =torch.tensor(val_X ,dtype=torch.float32)
val_Y = torch.tensor(val_Y ,dtype=torch.long)




train_loader = DataLoader(TensorDataset(train_X , train_Y), batch_size=64 ,shuffle=True)
test_loader = DataLoader(TensorDataset(test_X , test_Y), batch_size=64,shuffle=False)
val_loader = DataLoader(TensorDataset(val_X , val_Y), batch_size=64,shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mymodelMLP (nn.Module):
    def __init__(self , input_dim, hidden_1 , hidden_2):
        super().__init__()

        self.layer1 = nn.Linear(input_dim , hidden_1)
        self.layer2 = nn.Linear(hidden_1 , hidden_2)
        self.layer3 = nn.Linear(hidden_2 , hidden_1)
        self.relu = nn.ReLU()
        self.layer4 = nn.Linear(hidden_1 , 1)
        self.dropout = nn.Dropout(0.3)


    def forward (self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer4(x)

        return x
    

    
def objectiv (trial):
    lr = trial.suggest_float('lr' , 1e-4 , 1e-2 , log=True)
    hidden_1 = trial.suggest_int('hidden_1' , 16 , 64 , step=16)
    hidden_2 = trial.suggest_int('hidden_2' , 32 , 128 , step=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mymodelMLP(input_dim=train_X.shape[1] , hidden_1=hidden_1 , hidden_2=hidden_2)
    model.to(device)
    criterion = lambda inputs , targets :sigmoid_focal_loss(inputs , targets , alpha=0.85 , gamma=2.0 , reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    for i in range(epochs):

        model.train()

        for batch_X , batch_Y in train_loader :

            batch_X , batch_Y =   batch_X.to(device) , batch_Y.to(device)

            optimizer.zero_grad()

            logics = model(batch_X).squeeze()

            loss = criterion(logics , batch_Y.float())

            loss.backward()

            optimizer.step()


    model.eval()
    
    y_true, y_prob = [], []

    with torch.no_grad():
        for xb , yb in val_loader :

            xb = xb.to(device)

            logics_out = model(xb).squeeze()

            proba = torch.sigmoid(logics_out).cpu().numpy()

            y_prob.extend(proba)
            y_true.extend(yb.cpu().numpy())

            y_prds = (np.array(y_prob) > 0.5).astype(int)

            recall = recall_score(y_true , y_prds)

            return recall
        

sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objectiv, n_trials=50, show_progress_bar=True ,n_jobs=-1)

best_params = study.best_params
print("Best params:", best_params)

fin_model = mymodelMLP(input_dim=train_X.shape[1] , hidden_1=best_params['hidden_1'] , hidden_2=best_params['hidden_2'])
criterion = lambda inputs , targets : sigmoid_focal_loss(inputs , targets , alpha=0.85 ,gamma=2.0 , reduction='mean')
optimizers = optim.Adam(fin_model.parameters() , lr=best_params['lr'])

epochsis = 100
for i in range(epochsis):
    fin_model.train()

    for batch_X , batch_Y in train_loader :
        batch_X , batch_Y = batch_X.to(device) , batch_Y.to(device)

        logic_fin = fin_model(batch_X).squeeze()

        loss = criterion(logic_fin , batch_Y.float())

        loss.backward()
        optimizers.step()

        if (i+1) % 20 == 0 :
            print(f"Epoch {i+1}/{epochsis} - Loss: {loss.item():.4f}")


def evaluate_with_threshold(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).squeeze()
            prob = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(prob)
            y_true.extend(yb.cpu().numpy())
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr = thr[best_idx] if len(thr) > 0 else 0.5
    y_pred = (np.array(y_prob) >= best_thr).astype(int)
    return {
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred),
        'threshold': best_thr
    }

test_res = evaluate_with_threshold(fin_model, test_loader, device)
print(f"\nThreshold: {test_res['threshold']:.3f}")
print(f"Recall: {test_res['recall']:.4f}")
print(f"Precision: {test_res['precision']:.4f}")
print(f"F1 Score: {test_res['f1']:.4f}")

#________________

# Threshold: 0.556
# Recall: 0.9978
# Precision: 0.1525
# F1 Score: 0.2645

#_______________


def evaluaion_with_fixed_recall (model , loader , deviecs , target_recall=0.95):

    model.eval()
    y_true , y_proba = [] , []

    with torch.no_grad():
        for xb , yb in loader :
            xb = xb.to(deviecs)

            logicsd = model(xb).squeeze()

            proba = torch.sigmoid(logicsd).cpu().numpy()

            y_proba.extend(proba)
            y_true.extend(yb.cpu().numpy())

    
    thr = np.linspace(0.01 , 0.99 , 99)
    best_precision = 0
    best_thr = 0.5
    best_recall = 0



    for thrsl in thr :
        y_prds = (np.array(y_proba >= thrsl).astype(int))
        recall = recall_score(y_true , y_prds)
        precisions = precision_score(y_true , y_prds , zero_division= 0)


        if recall >= target_recall and precisions > best_precision :
            best_precision = precisions
            best_thr = thrsl
            best_recall = recall

    
    if best_precision == 0 :
        
        prec , rec , thr_test = precision_recall_curve(y_true , y_proba)

        f1_scored = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)

        best_idx = np.argmax(f1_scored)
        best_thr = thr_test[best_idx] if len(thr) > 0 else 0.5
        y_prds = (np.array(y_proba >= best_thr).astype(int))
        best_recall = recall_score(y_true , y_prds)
        best_precision = precision_score(y_true , y_prds , zero_division= 0)

    y_prds_finall = (np.array(y_proba >= best_thr).astype(int))

    return {
        'recall': best_recall,
        'precision': best_precision,
        'f1': f1_score(y_true, y_prds_finall),
        'auc': roc_auc_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_prds_finall),
        'threshold': best_thr,
        'y_true': y_true,
        'y_pred': y_prds_finall,
        'y_prob': y_proba
    }

# ارزیابی نهایی با شرط Recall >= 0.95
test_results =evaluaion_with_fixed_recall (fin_model, test_loader, device, target_recall=0.95)

print("\n" + "="*50)
print(f" Recall >= 0.95")
print("="*50)
print(f"best threshold:{test_results['threshold']:.3f}")
print(f"Accuracy : {test_results['accuracy']:.4f}")
print(f"Recall   : {test_results['recall']:.4f}")
print(f"Precision: {test_results['precision']:.4f}")
print(f"F1 Score : {test_results['f1']:.4f}")
print(f"AUC      : {test_results['auc']:.4f}")

print("\n Classification Report:")
print(classification_report(test_results['y_true'], test_results['y_pred'], target_names=['No (0)', 'Yes (1)']))


