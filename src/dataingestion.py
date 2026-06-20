import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/refs/heads/main/tweet_emotions.csv")

df  = df.drop(columns=['tweet_id'])

# filtering only happiness and sadness sentimnet from data set
finaldf = df[df["sentiment"].isin(["happiness", "sadness"])].copy()


# Assisgning binary 1 and 0 for prediction because we can use direct text
finaldf["sentiment"] = finaldf["sentiment"].map({'happiness':1, "sadness":0})


# diciding data into training and testig set 
train_data, test_data = train_test_split(finaldf, test_size=0.2, random_state=42)


# local Copy 
data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)
train_data.to_csv(os.path.join(data_path, "train.csv"), index = False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index = False)