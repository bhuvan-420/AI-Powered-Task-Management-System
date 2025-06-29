import pandas as pd
import numpy as np
import os

# Load cleaned data
df = pd.read_csv("data/processed/task_management_nlp_preprocessed.csv")

# Create a working copy to assign tasks
users = df[["assigned_to", "user_current_workload", "user_experience_level"]].drop_duplicates()
users.set_index("assigned_to", inplace=True)

# Normalize workload and experience for scoring
users["norm_workload"] = (users["user_current_workload"] - users["user_current_workload"].min()) / \
                         (users["user_current_workload"].max() - users["user_current_workload"].min())

users["norm_experience"] = (users["user_experience_level"] - users["user_experience_level"].min()) / \
                           (users["user_experience_level"].max() - users["user_experience_level"].min())

# Heuristic weights
weight_w = 0.6  # workload
weight_e = 0.4  # experience

# Shuffle tasks to simulate real assignment
task_df = df.copy().sample(frac=1, random_state=42).reset_index(drop=True)
task_df["new_assigned_to"] = ""

# Assign each task
for idx, row in task_df.iterrows():
    scores = users["norm_workload"] * weight_w - users["norm_experience"] * weight_e
    best_user = scores.idxmin()

    task_df.at[idx, "new_assigned_to"] = best_user
    users.at[best_user, "user_current_workload"] += row["estimated_hours"]

    # Update normalized workload
    w = users["user_current_workload"]
    users["norm_workload"] = (w - w.min()) / (w.max() - w.min())

# Save result
os.makedirs("data/processed", exist_ok=True)
task_df.to_csv("data/processed/task_balanced_assignments.csv", index=False)
print("✅ Balanced task assignments saved to: data/processed/task_balanced_assignments.csv")
