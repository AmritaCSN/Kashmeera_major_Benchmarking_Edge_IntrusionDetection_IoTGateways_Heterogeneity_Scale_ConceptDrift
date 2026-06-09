"""
train_baseline.py
=================
Trains RF and XGBoost baseline models on the REA-HID dataset.

FIX vs March version:
  - Added bidirectional_duration_ms, src2dst_duration_ms,
    dst2src_duration_ms to the drop list. These are derived directly
    from first_seen/last_seen timestamps and would allow the model to
    distinguish Sparse attacks (120-300s flows) from Benign (30-60s flows)
    using a timing artifact rather than behavioral pattern.
  - Added bidirectional_first_seen_ms, bidirectional_last_seen_ms
    (absolute timestamps — pure data leakage).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

print("[*] Loading REA-HID Publication Dataset...")
df = pd.read_csv("REA_HID_Publication_Dataset_v2.csv")

y = df['label']

# ── DROP LIST ────────────────────────────────────────────────
# Identifiers: IP, MAC, port, application tags, row id
# Temporal leakage: absolute timestamps, flow duration
#   (duration = last_seen - first_seen, directly encodes attack timing)
DROP = [
    # identifiers
    'label', 'attack_type', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
    'src_oui', 'dst_oui', 'application_name', 'application_category_name',
    'id', 'src_port', 'dst_port',
    # timestamp leakage
    'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
    'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
    'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
    # duration leakage (derived from timestamps)
    'bidirectional_duration_ms', 'src2dst_duration_ms', 'dst2src_duration_ms',
]

X = df.drop(columns=[c for c in DROP if c in df.columns], errors='ignore')
X = X.select_dtypes(include=[np.number])
X = X.fillna(0).replace([np.inf, -np.inf], 0)

print(f"[*] Features after leakage removal: {X.shape[1]}")
print(f"[*] Samples: {len(df)} | Benign: {(y==0).sum()} | Attack: {(y==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[*] Train: {len(X_train)} | Test: {len(X_test)}\n")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost":       XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                   random_state=42, n_jobs=-1)
}

for name, model in models.items():
    print(f"{'='*50}")
    print(f"Evaluating {name}...")
    print(f"{'='*50}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr      = fp / (fp + tn) if (fp + tn) > 0 else 0
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    print(f"Confusion Matrix:\n{cm}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"FPR:       {fpr:.4f}  ({fpr*100:.2f}%)\n")

print("[*] Baseline evaluation complete.")
print("[*] Next: python prove_evasion.py")
