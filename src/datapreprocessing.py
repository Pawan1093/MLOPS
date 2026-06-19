import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Load data
train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")

# Preprocess text column
train_df["content"] = train_df["content"].apply(preprocess_text)
test_df["content"] = test_df["content"].apply(preprocess_text)

data_path = os.path.join("data", "preprocessed")
os.makedirs(data_path, exist_ok=True)
train_df.to_csv(os.path.join(data_path, "train_processed.csv"))
test_df.to_csv(os.path.join(data_path, "test_processed.csv"))