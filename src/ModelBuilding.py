import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle

train_data = pd.read_csv("data/Featurestored/trainBow.csv")

X_train = train_data.iloc[:,0:-1].values
Y_train = train_data.iloc[:, -1].values

clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train, Y_train)

pickle.dump(clf, open('model.pkl', 'wb'))
