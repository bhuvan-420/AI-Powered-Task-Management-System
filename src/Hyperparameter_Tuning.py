import numpy as np
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Part A: Naive Bayes (Task Classification):

# Load data
df = pd.read_csv("data/processed/task_management_nlp_preprocessed.csv")
X = np.load("data/features/tfidf_features.npy")
y = LabelEncoder().fit_transform(df["category"])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    "alpha": [0.1, 0.5, 1.0],
    "fit_prior": [True, False]
}

grid = GridSearchCV(MultinomialNB(), param_grid, cv=3, scoring="f1_macro")
grid.fit(X_train, y_train)

# Best model
best_nb = grid.best_estimator_
y_pred = best_nb.predict(X_test)

print("📊 Tuned Naive Bayes Report:")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(best_nb, "results/models/task_classifier_nb_tuned.joblib")
print("💾 Saved: results/models/task_classifier_nb_tuned.joblib")
print("✅ Best Params:", grid.best_params_)


 # Part B: XGBoost (Priority Prediction):

# Load
df = pd.read_csv("data/processed/task_management_nlp_preprocessed.csv")
X = np.load("data/features/tfidf_features.npy")
y = LabelEncoder().fit_transform(df["priority"])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Grid
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.3]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
grid = GridSearchCV(xgb, param_grid, cv=3, scoring="f1_macro")
grid.fit(X_train, y_train)

# Best model
best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)

print("📊 Tuned XGBoost Report:")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(best_xgb, "results/models/priority_predictor_xgb_tuned.joblib")
print("💾 Saved: results/models/priority_predictor_xgb_tuned.joblib")
print("✅ Best Params:", grid.best_params_)