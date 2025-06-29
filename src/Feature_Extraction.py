import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib

# Load preprocessed descriptions
df = pd.read_csv("data/processed/task_management_nlp_preprocessed.csv")
corpus = df["description_clean"].astype(str).tolist()

# TF-IDF setup
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words="english"
)

print("🔢 Extracting TF-IDF features (unigrams + bigrams)...")
tfidf_matrix = vectorizer.fit_transform(corpus)
features = tfidf_matrix.toarray()

# Save features

joblib.dump(vectorizer, "results/models/tfidf_vectorizer.joblib")

os.makedirs("data/features", exist_ok=True)
np.save("data/features/tfidf_1000_bigrams.npy", features)
print(f"✅ Saved: data/features/tfidf_1000_bigrams.npy — shape: {features.shape}")
