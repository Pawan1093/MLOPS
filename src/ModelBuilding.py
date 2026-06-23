import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml

with open("params.yaml", 'r') as file:
    params1 = yaml.safe_load(file)

n_estimator = params1["ModelBuilding"]["n_estimators"]


train_data = pd.read_csv("data/Featurestored/trainBow.csv")

X_train = train_data.iloc[:,0:-1].values
Y_train = train_data.iloc[:, -1].values

clf = GradientBoostingClassifier(n_estimators=n_estimator)
clf.fit(X_train, Y_train)

pickle.dump(clf, open('model.pkl', 'wb'))
