import pandas as pd 
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

# featuching preprocessed data 
train_data  = pd.read_csv("data/preprocessed/train_processed.csv")
test_data = pd.read_csv("data/preprocessed/test_processed.csv")

X_train = train_data["content"].fillna("").astype(str)
Y_train = train_data["sentiment"] 

X_test = test_data["content"].fillna("").astype(str)
Y_test = test_data["sentiment"]



# bag of Words
vectorizer = CountVectorizer()

X_trainBow = vectorizer.fit_transform(X_train)
X_testBow = vectorizer.transform(X_test)



train_df = pd.DataFrame(X_trainBow.toarray())
train_df ['label'] = Y_train

test_df = pd.DataFrame(X_trainBow.toarray())
test_df ['label'] = Y_train

data_path = os.path.join("data", "Featurestored")
os.makedirs(data_path, exist_ok=True)
train_df.to_csv(os.path.join(data_path, "trainBow.csv"), index = False)
test_df.to_csv(os.path.join(data_path, "testBow.csv"), index= False)