import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

np.random.seed(42)
random.seed(42)

n_rows = 1000

# Categories and users
categories = ['Development', 'Testing', 'Design', 'Research', 'Marketing']
priorities = ['Low', 'Medium', 'High', 'Critical']
statuses = ['Not Started', 'In Progress', 'Completed']
departments = ['Engineering', 'Marketing', 'Sales']
users = [f"user_{i:03d}" for i in range(1, 11)]

priority_templates = {
    'Low':      ["Routine task: {}", "Optional update: {}", "Regular cleanup: {}", "Periodic activity: {}"],
    'Medium':   ["Standard task: {}", "Scheduled work: {}", "Normal process: {}", "Assigned action: {}"],
    'High':     ["Important task: {}", "Time-sensitive: {}", "Handle ASAP: {}", "High priority item: {}"],
    'Critical': ["🚨 Urgent: {}", "‼️ Immediate attention: {}", "🔥 Critical issue: {}", "Priority 1 task: {}"]
}

subjects = [
    'user authentication', 'data migration', 'report generation',
    'dashboard optimization', 'system performance', 'security audit',
    'content publishing', 'sales funnel', 'workflow automation'
]

def generate_task_description(priority):
    template = random.choice(priority_templates[priority])
    subject = random.choice(subjects)
    return template.format(subject)

def generate_dates():
    start = datetime.now() - timedelta(days=180)
    created = start + timedelta(days=random.randint(0, 180))
    due = created + timedelta(days=random.randint(1, 15))
    return created, due

# Create tasks
rows = []
for i in range(n_rows):
    priority = priorities[i % 4]  # ensure balance
    category = random.choice(categories)
    description = generate_task_description(priority)
    created, due = generate_dates()

    rows.append({
        "task_id": f"TASK_{i+1:05d}",
        "title": description.split(":")[0],
        "description": description,
        "category": category,
        "priority": priority,
        "status": random.choice(statuses),
        "created_date": created.strftime("%Y-%m-%d"),
        "due_date": due.strftime("%Y-%m-%d"),
        "completion_date": "",
        "assigned_to": random.choice(users),
        "assigned_to_name": f"User_{random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'])}",
        "created_by": "admin",
        "created_by_name": "System",
        "department": random.choice(departments),
        "estimated_hours": round(np.random.uniform(3, 15), 2),
        "actual_hours": "",
        "complexity_score": round(np.random.uniform(1, 10), 1),
        "dependencies_count": random.randint(0, 5),
        "user_current_workload": random.randint(1, 12),
        "user_experience_level": round(np.random.uniform(1, 10), 1),
        "task_age_days": random.randint(1, 90),
        "is_overdue": random.choice([0, 1])
    })

df = pd.DataFrame(rows)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/task_management_dataset.csv", index=False)
print("✅ Generated: data/raw/task_management_dataset_priority_signal.csv")