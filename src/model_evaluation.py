import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score


# 1. Load the model and data
model = pickle.load(open('model.pkl', 'rb'))
test_df = pd.read_csv("data/featurestored/testBow.csv")

# 2. Separate Features and Labels
X_test = test_df.iloc[:,0:-1].values
Y_test = test_df.iloc[:, -1].values

# 3. Get predictions and probabilities
# predict() gives you 0 or 1
y_pred = model.predict(X_test)

# predict_proba() gives you the probability of each class [prob_0, prob_1]
# We take the column at index 1 for the positive class (happiness)
y_prob = model.predict_proba(X_test)[:, 1]

# 4. Calculate Metrics
print(f"Accuracy: {accuracy_score(Y_test, y_pred):.2f}")
print(f"ROC-AUC Score: {roc_auc_score(Y_test, y_prob):.2f}")

#metric 

import json


metrics = {
    "accuracy": accuracy_score(Y_test, y_pred),
    "roc_auc": roc_auc_score(Y_test, y_prob),
    "precison":precision_score(Y_test, y_pred)
}

# Save metrics to a file
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

