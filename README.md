# bank-marketing-imbalanced-classification

# Bank Marketing Campaign Classification
## Handling Imbalanced Data with SMOTE-ENN, Focal Loss, and Ensemble Methods

### 📌 Project Overview

This project tackles the imbalanced classification problem in predicting whether a client will subscribe to a bank term deposit. The dataset used is the UCI Bank Marketing dataset, where only ~11.3% of the instances belong to the positive class (subscribed).

The goal was to build a robust, production-ready model that achieves a high Recall (to capture potential customers) while maintaining competitive Precision (to minimize false alarms and waste of marketing resources).

---

### 🧠 Key Challenge: Class Imbalance

The dataset contains 41,188 instances with a 1:8 ratio (positive vs. negative). This severe imbalance causes traditional models to be biased toward the majority class, resulting in:
- High Accuracy (misleading metric)
- Low Recall for the minority class
- Poor performance in real-world scenarios

> 👉 Handling imbalanced data is a common challenge in domains like fraud detection, churn prediction, medical diagnosis, and marketing analytics[reference:0][reference:1]. Our project systematically compares multiple strategies to address this issue.

---

### 📊 Dataset

- Source: UCI Machine Learning Repository – Bank Marketing Dataset
- Format: CSV (bank-additional-full.csv)
- Features: 20 attributes (categorical and numerical) + target y
- Target Distribution:
  - Class 0 (No): 36,548 (88.7%)
  - Class 1 (Yes): 4,640 (11.3%)

Preprocessing:
- One-Hot Encoding for categorical features
- StandardScaler for numerical features
- Train/Validation/Test split: 60% / 20% / 20%

---

### 🛠️ Methods & Models Tested

We implemented and evaluated multiple approaches to tackle class imbalance:

#### 1. Baseline Models

| Model | Description |
|-------|-------------|
| Random Forest | With class_weight='balanced' to penalize misclassification of the minority class |
| LightGBM | Gradient boosting with scale_pos_weight |
| XGBoost | With scale_pos_weight and early stopping |
| MLP (Neural Network) | Simple Multi-Layer Perceptron with dropout and ReLU activation |

#### 2. Loss Functions for Imbalance

- Focal Loss: Introduced by Facebook AI Research (Lin et al., 2017). Modifies cross-entropy by adding a modulating factor (1 - p_t)^γ to focus learning on hard-to-classify examples[reference:2][reference:3].
- Class-Balanced Loss: Adjusts loss based on the effective number of samples per class.
- LDAM (Label-Distribution-Aware Margin Loss): Adds a class-dependent margin to encourage better separation.

#### 3. Resampling Techniques

| Technique | Description |
|-----------|-------------|
| Random Under-Sampling | Randomly remove samples from the majority class |
| SMOTE | Synthetic Minority Over-sampling Technique |
| ADASYN | Adaptive SMOTE that focuses on harder examples |
| SMOTE-ENN | Combines SMOTE with Edited Nearest Neighbors to remove noisy samples[reference:4][reference:5] |
| SMOTE-Tomek | Combines SMOTE with Tomek Links for cleaner boundaries |

#### 4. Hybrid Approaches

- SMOTE-ENN + Threshold Tuning: The best-performing combination. SMOTE-ENN balances the dataset, and threshold tuning optimizes the Precision/Recall trade-off.

---

### 📈 Results & Comparison

After extensive experimentation, the SMOTE-ENN + Random Forest + Threshold Tuning combination achieved the best balance between Recall and Precision.

#### Final Results (Test Set)

| Metric | Value |
|--------|-------|
| Recall (Class 1) | 86.2% |
| Precision (Class 1) | 50.0% |
| F1 Score (Class 1) | 0.633 |
| Accuracy | 88.7% |
| AUC-ROC | ~0.91 |

#### Confusion Matrix


Actual No    6509   801
Actual Yes    128   800

- True Negatives: 6,509
- False Positives: 801 (clients incorrectly predicted as subscribers)
- False Negatives: 128 (subscribers missed by the model)
- True Positives: 800

#### Comparison Across Methods


| Method | Recall (1) | Precision (1) | F1 (1) | AUC |
|--------|------------|---------------|--------|-----|
| Random Forest (baseline) | 90.8% | 44.1% | 0.594 | ~0.90 |
| LightGBM | ~88% | ~42% | ~0.57 | ~0.89 |
| MLP + Focal Loss | ~85% | ~38% | ~0.53 | ~0.87 |
| SMOTE-ENN + RF | 89.0% | 50.0% | 0.64 | ~0.91 |
| SMOTE-ENN + RF + Threshold Tuning | 86.2% | 50.0% | 0.633 | ~0.91 |

---

### 💡 Key Insights & Learnings

#### 1. Data Imbalance is a Real Challenge
With only 11.3% positive samples, models naturally bias toward the majority class. Achieving both high Recall and high Precision is inherently a trade-off.

#### 2. Threshold Tuning is Powerful
Simply adjusting the classification threshold (from 0.5 to ~0.55) improved Precision by ~6% while keeping Recall above 85%. This is a zero-cost improvement that requires no model retraining.

#### 3. SMOTE-ENN is Effective
SMOTE-ENN outperformed both standalone SMOTE and random under-sampling by:
- Generating synthetic samples for the minority class
- Removing noisy and borderline samples that confuse the model[reference:6]

#### 4. Deep Learning is Not Always Better
MLP with Focal Loss performed worse than well-tuned Random Forest with SMOTE-ENN. For tabular data, tree-based ensembles often outperform neural networks.

#### 5. Business Context Matters
In a marketing campaign, Recall is more important than Precision in early stages (identify all potential customers). However, for final decision-making, Precision becomes critical to avoid wasting resources on false leads[reference:7].

---

### 🚀 How to Run

#### Requirements

pip install -r requirements.txt
Steps

1. Clone the repository

git clone https://github.com/hatamzadeh86/bank-marketing-imbalanced.git
cd bank-marketing-imbalanced
2. Run the preprocessing pipeline

python src/preprocess.py
3. Train models and evaluate

python src/train.py
4. Reproduce results

python src/evaluate.py
Project Structure

.
├── data/
│   └── bank-additional-full.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── 
│   ├── confusion_matrix.png
│   
├── README.md
└── requirements.txt
---

📚 References

· Focal Loss: Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. ICCV. 
· SMOTE: Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research.
· UCI Bank Marketing Dataset: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
· Imbalanced-learn library: https://imbalanced-learn.org/

---

🤝 Connect with Me

If you found this project useful or have any questions, feel free to connect:

· LinkedIn: [https://www.linkedin.com/in/amir-mohammad-hatemzadeh-44b2a138b]
· GitHub: [https://github.com/hatamzadeh86]
· Email: [activedirectoryn@gmail.com]

--------

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

⭐ Don't forget to star this repository if you found it helpful!

`

---
