import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Load cleaned data
df = pd.read_csv("data/processed/task_management_cleaned.csv")

# Setup lemmatizer
lemmatizer = WordNetLemmatizer()

# Define improved preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [w for w in tokens if w.isalpha() and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Apply preprocessing
print("🔄 Preprocessing task descriptions...")
df["description_clean"] = df["description"].apply(preprocess_text)

# Preview results
print("\n📌 Sample cleaned descriptions:")
print(df[["description", "description_clean"]].head())

# Save output
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/task_management_nlp_preprocessed.csv", index=False)
print("✅ NLP-preprocessed data saved to: data/processed/task_management_nlp_preprocessed.csv")
