import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/raw/task_management_dataset.csv")  # ✅ Adjust file name if needed
print(f"📦 Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------
# 🔍 Initial Checks
# ---------------------
print("\n🧹 Checking for missing values...")
print(df.isnull().sum())

print("\n🔁 Checking for duplicates...")
duplicates = df.duplicated().sum()
print(f"Found {duplicates} duplicates")

# ---------------------
# 📊 Class Balance Visuals
# ---------------------
os.makedirs("results/eda", exist_ok=True)

# Priority distribution (always available)
plt.figure(figsize=(10, 5))
sns.countplot(x="priority", data=df, order=["Critical", "High", "Medium", "Low"], palette="viridis")
plt.title("Task Priority Distribution")
plt.xlabel("Priority")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/eda/priority_distribution.png")
plt.close()

# Category distribution (safe check)
if "category" in df.columns and df["category"].nunique() > 1:
    plt.figure(figsize=(10, 5))
    sns.countplot(x="category", data=df, palette="Set2")
    plt.title("Task Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/eda/category_distribution.png")
    plt.close()
else:
    print("⚠️ Skipping category plot — not enough variation or missing column.")

# ---------------------
# 🔍 Description Cleanup
# ---------------------
df["description"] = df["description"].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

# ---------------------
# 💾 Save Cleaned Data
# ---------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/task_management_cleaned.csv", index=False)
print("\n✅ Cleaned data saved to: data/processed/task_management_cleaned.csv")