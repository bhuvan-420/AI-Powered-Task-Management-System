import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load text features and data
df = pd.read_csv("data/processed/task_management_nlp_preprocessed.csv")
X_tfidf = np.load("data/features/tfidf_1000_bigrams.npy")

# Label encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["category"])

# Select numeric structured features
structured_features = df[[
    "complexity_score",
    "estimated_hours",
    "user_experience_level",
    "user_current_workload"
]]

# Optional: encode department (categorical)
dept_encoded = pd.get_dummies(df["department"], prefix="dept")

# Combine all
X_struct = pd.concat([structured_features, dept_encoded], axis=1)
scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_struct)

# Combine TF-IDF + structured
X_full = np.hstack([X_tfidf, X_struct_scaled])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, stratify=y, test_size=0.2, random_state=42
)

# RandomizedSearchCV for XGBoost
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.3],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1_macro",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("🔍 Tuning XGBoost on text + structured features...")
search.fit(X_train, y_train)

# Evaluate
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📊 Final Classification Report (XGBoost + Fusion):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("✅ Best Params:", search.best_params_)

# Save
os.makedirs("results/models", exist_ok=True)
joblib.dump(best_model, "results/models/task_classifier_xgb_fusion.joblib")
print("💾 Saved: task_classifier_xgb_fusion.joblib")