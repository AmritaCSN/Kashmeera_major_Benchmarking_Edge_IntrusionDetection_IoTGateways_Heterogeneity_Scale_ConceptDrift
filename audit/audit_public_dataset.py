import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading Public MQTTEEB-D Cleaned Dataset...")
file_path = "data/raw/MQTTEEB-D_Final_Dataset/Preprocessed_Data/MQTTEEB-D_cleaned_data.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    file_path = "../" + file_path
    df = pd.read_csv(file_path)

# Automatically identify the label column
label_col = [col for col in df.columns if col.lower() in ['label', 'class', 'target', 'attack_type']][0]
print(f"Found target column: {label_col}")

# --- FIX: Drop any rows where the label is NaN before doing anything else ---
initial_len = len(df)
df = df.dropna(subset=[label_col])
print(f"Dropped {initial_len - len(df)} rows with missing labels.")

# Separate features and target
X = df.drop(columns=[label_col])
y = df[label_col]

# Keep only numeric features and clean them
X = X.select_dtypes(include=[np.number])
X = X.fillna(0).replace([np.inf, -np.inf], 0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Training instances: {len(X_train)} | Testing instances: {len(X_test)}")

print("\nTraining Random Forest on Public Dataset...")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\n" + "="*40)
print("Public Dataset Evaluation Results")
print("="*40)
print(classification_report(y_test, y_pred))

# --- Feature Importance Extraction & Plotting ---
print("\nExtracting Feature Importances...")
importances = rf.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

# Generate the plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax)

ax.set_title('Top 10 Features Used by Model on Public Dataset (MQTTEEB-D)', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('Relative Importance (Gini Importance)', fontsize=12)
ax.set_ylabel('Network Feature', fontsize=12)

# Add a text box explaining the flaw
textstr = "Notice the dominance of Volumetric features\n(sizes, counts) vs Temporal features (IAT, timing).\nThis causes catastrophic failure against\ntiming-aware low-and-slow attacks."
props = dict(boxstyle='round', facecolor='#ff9896', alpha=0.9)
ax.text(0.40, 0.50, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', bbox=props)

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('public_feature_flaw.png', dpi=300)
print("\nGraph saved as 'public_feature_flaw.png'")