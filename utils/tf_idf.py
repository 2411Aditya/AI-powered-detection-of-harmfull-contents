import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from layer1_prep import preprocess

# Load dataset
df = pd.read_csv(r'C:\Users\Aditya Kulkarni\Desktop\Projects\Harmful content detection\data\Layer_1_data.csv')

# Convert to binary labels: 0 = harmful, 1 = clean
df['label'] = df['class'].apply(lambda x: 1 if x == 2 else 0)

# Preprocess the tweets
df['clean_tweet'] = df['tweet'].apply(preprocess)

# Features and labels
X = df['clean_tweet']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)