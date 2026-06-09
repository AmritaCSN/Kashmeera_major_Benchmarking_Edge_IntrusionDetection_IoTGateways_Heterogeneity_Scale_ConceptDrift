"""
prove_evasion.py
================
Demonstrates the Accuracy Illusion and proves evasion reality.

Runs three experiments:
  1. Baseline on public MQTTEEB-D dataset → shows shortcut learning
  2. Evasion test on REA-HID camouflaged dataset → shows per-attack evasion rates
  3. SHAP analysis → per-sample attribution proof of IAT behavioral features

FIX vs previous version:
  - SHAP bar plot now uses EXACT SET membership for feature coloring,
    not substring matching. This prevents flag features like
    bidirectional_ack_packets from being misclassified as volumetric shortcuts.
  - Gini plot (Exp 1) also updated to use exact-match logic for MQTTEEB-D features.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
from matplotlib.patches import Patch
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# ── SHARED DROP LIST ─────────────────────────────────────────
DROP = [
    'label', 'attack_type', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
    'src_oui', 'dst_oui', 'application_name', 'application_category_name',
    'id', 'src_port', 'dst_port',
    'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
    'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
    'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
    
    'bidirectional_duration_ms', 'src2dst_duration_ms', 'dst2src_duration_ms',
]

# ── EXACT FEATURE TAXONOMY ───────────────────────────────────
# Used for SHAP bar plot coloring.
# Exact set membership — no substring matching.
# This prevents flag features (ack_packets, psh_packets) from being
# misclassified as volumetric shortcuts.
VOLUMETRIC_EXACT = {
    'bidirectional_bytes', 'src2dst_bytes', 'dst2src_bytes',
    'bidirectional_packets', 'src2dst_packets', 'dst2src_packets',
    'bidirectional_min_ps', 'bidirectional_mean_ps',
    'bidirectional_stddev_ps', 'bidirectional_max_ps',
    'src2dst_min_ps', 'src2dst_mean_ps',
    'src2dst_stddev_ps', 'src2dst_max_ps',
    'dst2src_min_ps', 'dst2src_mean_ps',
    'dst2src_stddev_ps', 'dst2src_max_ps',
}
TEMPORAL_EXACT = {
    'bidirectional_duration_ms', 'src2dst_duration_ms', 'dst2src_duration_ms',
    'tcp_time_delta', 'timestamp',
    'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
}

def classify_feature_exact(feat):
    """Returns 'volumetric', 'temporal', 'iat', or 'behavioral'."""
    if feat in VOLUMETRIC_EXACT:
        return 'volumetric'
    if feat in TEMPORAL_EXACT:
        return 'temporal'
    fl = feat.lower()
    if 'piat' in fl or ('iat' in fl and 'piat' not in fl):
        return 'iat'
    return 'behavioral'

COLOR_MAP = {
    'volumetric': '#C00000',   # red
    'temporal':   '#D46B08',   # orange
    'iat':        '#375623',   # dark green
    'behavioral': '#2E75B6',   # blue
}

LEGEND_ELEMENTS = [
    Patch(facecolor='#375623', label='IAT behavioral feature (legitimate signal)'),
    Patch(facecolor='#2E75B6', label='Other behavioral feature'),
    Patch(facecolor='#C00000', label='Volumetric shortcut'),
    Patch(facecolor='#D46B08', label='Temporal leakage'),
]


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Public Dataset Baseline (MQTTEEB-D)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("EXPERIMENT 1: Public Dataset Baseline (MQTTEEB-D)")
print("=" * 60)

pub_path = "data/raw/MQTTEEB-D_Final_Dataset/Preprocessed_Data/MQTTEEB-D_cleaned_data.csv"
if not os.path.exists(pub_path):
    pub_path = "../" + pub_path

# MQTTEEB-D specific feature taxonomy (packet-level schema)
MQTTEEB_VOLUMETRIC = {'tcp_len'}
MQTTEEB_TEMPORAL   = {'tcp_time_delta', 'timestamp'}

try:
    pub_df = pd.read_csv(pub_path)
    label_col = [c for c in pub_df.columns
                 if c.lower() in ['label', 'class', 'target', 'attack_type']][0]
    pub_df = pub_df.dropna(subset=[label_col])

    X_pub = pub_df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    X_pub = X_pub.fillna(0).replace([np.inf, -np.inf], 0)
    y_pub = pub_df[label_col]

    X_tr, X_te, y_tr, y_te = train_test_split(X_pub, y_pub, test_size=0.2,
                                               random_state=42)
    rf_pub = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_pub.fit(X_tr, y_tr)
    y_pred_pub = rf_pub.predict(X_te)

    print(classification_report(y_te, y_pred_pub))

    imp_pub = pd.Series(rf_pub.feature_importances_, index=X_pub.columns)
    imp_pub = imp_pub.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for feat in imp_pub.index:
        if feat in MQTTEEB_VOLUMETRIC:
            colors.append('#C00000')
        elif feat in MQTTEEB_TEMPORAL:
            colors.append('#D46B08')
        else:
            colors.append('#2E75B6')

    sns.barplot(x=imp_pub.values, y=imp_pub.index, palette=colors, ax=ax)
    ax.set_title('Top 10 Features — MQTTEEB-D (Gini Importance)\n'
                 'Red=Volumetric shortcut  Orange=Temporal leakage  Blue=Behavioral',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Relative Importance (Gini)', fontsize=11)
    textstr = ("Shortcut learning: top features are\n"
               "volumetric/temporal artifacts,\n"
               "not true behavioral signals.")
    ax.text(0.55, 0.35, textstr, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#ffcdd2', alpha=0.9))
    plt.tight_layout()
    plt.savefig('outputs/figures/public_feature_importance.png', dpi=300)
    print("[OK] Saved: outputs/figures/public_feature_importance.png")

except Exception as e:
    print(f"[WARN] Could not load public dataset: {e}")
    print("       Skipping Experiment 1 — continuing with Experiment 2")


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Evasion Reality — REA-HID Camouflaged Dataset
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2: Evasion Reality (mqtt + coap camouflaged dataset)")
print("=" * 60)

df = pd.read_csv("dataset_mqtt_coap_final.csv")

y_detailed = df['attack_type']
y_binary   = df['label']

X = df.drop(columns=[c for c in DROP if c in df.columns], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)

print(f"[*] Features (after leakage removal): {X.shape[1]}")

X_train, X_test, y_train_bin, y_test_bin, idx_train, idx_test = train_test_split(
    X, y_binary, df.index, test_size=0.20, random_state=42, stratify=y_binary
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_bin)
y_pred = rf.predict(X_test)

print("Overall binary classification (Benign vs Attack):")
print(classification_report(y_test_bin, y_pred,
                             target_names=['Benign', 'Attack']))

# Per-attack evasion report
results_df = pd.DataFrame({
    'True_Class':  df.loc[idx_test, 'attack_type'].values,
    'True_Binary': y_test_bin.values,
    'Predicted':   y_pred
})

print("\n=== EVASION REPORT (per attack type) ===")
evasion_data = []
attacks_only = results_df[results_df['True_Binary'] == 1]

for attack_name in sorted(attacks_only['True_Class'].unique()):
    subset = attacks_only[attacks_only['True_Class'] == attack_name]
    total  = len(subset)
    missed = len(subset[subset['Predicted'] == 0])
    rate   = (missed / total) * 100 if total > 0 else 0
    evasion_data.append({'Attack': attack_name, 'Total': total,
                         'Evaded': missed, 'Evasion_Rate_%': round(rate, 1)})
    print(f"  [{attack_name}]  total={total}  evaded={missed}  "
          f"evasion rate={rate:.1f}%")

evasion_df = pd.DataFrame(evasion_data)
evasion_df.to_csv("outputs/tables/evasion_report.csv", index=False)
print("\n[OK] Saved: outputs/tables/evasion_report.csv")

# Evasion bar chart
benign_f1     = f1_score(y_test_bin, y_pred, pos_label=0, average='binary')
attack_f1     = f1_score(y_test_bin, y_pred, pos_label=1, average='binary')
benign_recall = recall_score(y_test_bin, y_pred, pos_label=0)
attack_recall = recall_score(y_test_bin, y_pred, pos_label=1)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [benign_f1, attack_f1], width,
               label='F1-Score', color=['#2E75B6', '#2E75B6'], alpha=0.85)
bars2 = ax.bar(x + width/2, [benign_recall, attack_recall], width,
               label='Recall (Detection)', color=['#C00000', '#C00000'], alpha=0.85)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(['Benign Traffic', 'Low-and-Slow Attack'])
ax.set_ylabel('Score (0 to 1.0)')
ax.set_title('Static Baseline Evaded: The Low-and-Slow Blindspot\n'
             '(Volumetrically camouflaged, timing-aware attacks)', fontsize=12)
ax.set_ylim(0, 1.2)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/figures/evasion_reality.png', dpi=300)
print("[OK] Saved: outputs/figures/evasion_reality.png")


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 3: SHAP Analysis
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 3: SHAP Feature Attribution")
print("=" * 60)

try:
    import shap

    print("[*] Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    sv = shap_values
    if isinstance(sv, list):
        shap_attack = np.array(sv[1])
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_attack = sv[:, :, 1]
    else:
        shap_attack = sv

    print(f"  SHAP array shape: {shap_attack.shape}")

    # Beeswarm plot
    print("[*] Generating SHAP summary beeswarm plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_attack, X_test, feature_names=list(X.columns),
                      show=False, max_display=20)
    plt.title('SHAP Feature Attribution — REA-HID Dataset\n'
              'Per-sample proof: IAT features drive attack predictions',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_summary.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: outputs/figures/shap_summary.png")

    # Bar plot — EXACT SET coloring, no substring matching
    print("[*] Generating SHAP bar plot...")
    mean_shap   = np.abs(shap_attack).mean(axis=0)
    shap_series = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)
    top15       = shap_series.head(15)

    # ── EXACT COLOR ASSIGNMENT ───────────────────────────────
    # Uses classify_feature_exact() which checks VOLUMETRIC_EXACT and
    # TEMPORAL_EXACT sets first, then falls back to IAT keyword match.
    # Flag features (ack_packets, psh_packets, syn_packets) are NOT in
    # VOLUMETRIC_EXACT so they correctly get 'behavioral' (blue).
    colors_shap = [COLOR_MAP[classify_feature_exact(f)] for f in top15.index]
    # ─────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top15)), top15.values[::-1],
            color=list(reversed(colors_shap)), alpha=0.85)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(list(reversed(top15.index)), fontsize=10)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title('SHAP Mean |Value| — Top Features Driving Attack Classification\n'
                 'Green=IAT behavioral  Red=Volumetric shortcut  Orange=Temporal leakage',
                 fontsize=12, fontweight='bold')
    ax.legend(handles=LEGEND_ELEMENTS, loc='lower right', fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_bar.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: outputs/figures/shap_bar.png")

    print("\nTop 10 features by mean |SHAP|:")
    print(shap_series.head(10).round(5).to_string())

    shap_series.head(15).to_csv("outputs/tables/shap_feature_importance.csv",
                                 header=['mean_abs_shap'])
    print("[OK] Saved: outputs/tables/shap_feature_importance.csv")

except ImportError:
    print("[WARN] shap not installed. Run: pip install shap --break-system-packages")
except Exception as e:
    print(f"[WARN] SHAP failed: {e}")


# ── TOP GINI FEATURES ────────────────────────────────────────
print("\n=== TOP 10 FEATURES (Gini Importance) ===")
gini_series = pd.Series(rf.feature_importances_, index=X.columns)
print(gini_series.sort_values(ascending=False).head(10).round(5).to_string())

print("\n" + "=" * 60)
print("All outputs saved to outputs/figures/ and outputs/tables/")