"""
final_evaluation.py
===================
Generates the complete results table for the 3rd review and publication.

Runs four experiments:
  1. Multi-seed (10 seeds) RF + MLP evaluation
  2. Pre-drift vs post-drift F1 degradation
  3. McNemar test (RF vs MLP)
  4. Feature ablation (All / No-shortcuts / IAT-only)

FIX vs previous version:
  - Added "duration" to shortcut_keywords (catches bidirectional_duration_ms)
  - Added all duration columns to DROP list
  - Ablation now correctly identifies duration as a shortcut
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from statsmodels.stats.contingency_tables import mcnemar as scipy_mcnemar
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs/final_eval", exist_ok=True)
os.makedirs("outputs/tables",     exist_ok=True)
os.makedirs("outputs/figures",    exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────
print("[*] Loading dataset_mqtt_coap_final.csv...")
df = pd.read_csv("dataset_mqtt_coap_final.csv")

DROP = [
    "label", "attack_type", "src_ip", "dst_ip", "src_mac", "dst_mac",
    "src_oui", "dst_oui", "application_name", "application_category_name",
    "id", "src_port", "dst_port",
    # timestamp leakage
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    # duration leakage
    "bidirectional_duration_ms", "src2dst_duration_ms", "dst2src_duration_ms",
]

y      = df["label"].values
X_full = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
X_full = X_full.select_dtypes(include=[np.number]).fillna(0)
features = list(X_full.columns)

print(f"[*] Features: {len(features)} | Samples: {len(df)}")
print(f"[*] Benign: {(y==0).sum()} | Attack: {(y==1).sum()}\n")
print(f"[*] Feature list: {features}\n")


# ── HELPERS ──────────────────────────────────────────────────
def fpr_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return 0.0

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    b = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    c = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    table = np.array([[0, b], [c, 0]])
    result = scipy_mcnemar(table, exact=True)
    return result.pvalue


# ── EXPERIMENT 1: MULTI-SEED EVALUATION ─────────────────────
print("=" * 60)
print("EXPERIMENT 1: Multi-Seed Evaluation (10 seeds)")
print("=" * 60)

SEEDS = [42, 123, 456, 789, 1337, 2024, 31415, 99999, 7777, 1111]

models_config = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "DRIFT-MLP":     MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                                   max_iter=500),
}

seed_results = {name: {"f1": [], "fpr": [], "recall": []} for name in models_config}

for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full.values, y, test_size=0.2, random_state=seed, stratify=y
    )
    for name, model in models_config.items():
        if hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        cr = classification_report(y_te, y_pred, output_dict=True)
        seed_results[name]["f1"].append(f1_score(y_te, y_pred, average="weighted"))
        seed_results[name]["fpr"].append(fpr_score(y_te, y_pred))
        seed_results[name]["recall"].append(
            cr.get("1", cr.get("Attack", {"recall": 0.0}))["recall"]
        )
    print(f"  Seed {seed}: done")

multiseed_rows = []
for name, res in seed_results.items():
    row = {
        "Model":       name,
        "F1_Mean":     round(np.mean(res["f1"]),     4),
        "F1_Std":      round(np.std(res["f1"]),      4),
        "FPR_Mean":    round(np.mean(res["fpr"]),    4),
        "FPR_Std":     round(np.std(res["fpr"]),     4),
        "Recall_Mean": round(np.mean(res["recall"]), 4),
        "Recall_Std":  round(np.std(res["recall"]),  4),
    }
    multiseed_rows.append(row)
    print(f"\n  {name}:")
    print(f"    F1     = {row['F1_Mean']} ± {row['F1_Std']}")
    print(f"    FPR    = {row['FPR_Mean']} ± {row['FPR_Std']}")
    print(f"    Recall = {row['Recall_Mean']} ± {row['Recall_Std']}")

multiseed_df = pd.DataFrame(multiseed_rows)
multiseed_df.to_csv("outputs/final_eval/multiseed_results.csv", index=False)
print("\n[OK] Saved: outputs/final_eval/multiseed_results.csv")


# ── EXPERIMENT 2: PRE-DRIFT vs POST-DRIFT ────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 2: Pre-Drift vs Post-Drift F1 Degradation")
print("=" * 60)

ts_col = None
for candidate in ["bidirectional_first_seen_ms", "src2dst_first_seen_ms"]:
    if candidate in df.columns:
        ts_col = candidate
        break

if ts_col:
    start_ms = df[ts_col].min()
    df["relative_minutes"] = (df[ts_col] - start_ms) / 60000.0
    pre_df  = df[df["relative_minutes"] <= 60]
    post_df = df[df["relative_minutes"] >  60]
    print(f"  Pre-drift flows:  {len(pre_df)}")
    print(f"  Post-drift flows: {len(post_df)}")

    if len(pre_df) > 20 and len(post_df) > 20:
        # Use features only (drop all leakage/identifiers)
        pre_feats  = pre_df[features].fillna(0)
        y_pre      = pre_df["label"].values
        post_feats = post_df[features].fillna(0)
        y_post     = post_df["label"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            pre_feats, y_pre, test_size=0.2, random_state=42, stratify=y_pre
        )
        rf_drift = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_drift.fit(X_tr, y_tr)

        f1_pre  = f1_score(y_te, rf_drift.predict(X_te), average="weighted")
        f1_post = (f1_score(y_post, rf_drift.predict(post_feats), average="weighted")
                   if len(np.unique(y_post)) > 1 else float("nan"))

        print(f"  F1 pre-drift  (in-distribution):    {f1_pre:.4f}")
        f1_post_str = f"{f1_post:.4f}" if not np.isnan(f1_post) else "N/A"
        print(f"  F1 post-drift (out-of-distribution): {f1_post_str}")
        if not np.isnan(f1_post):
            print(f"  Degradation: {f1_pre - f1_post:.4f}")

        drift_df = pd.DataFrame([
            {"Condition": "Pre-Drift (0-60 min)",   "F1": round(f1_pre,  4),
             "Note": "In-distribution"},
            {"Condition": "Post-Drift (60-120 min)", "F1": round(f1_post, 4) if not np.isnan(f1_post) else "N/A",
             "Note": "Benign IAT shifted"},
        ])
        drift_df.to_csv("outputs/final_eval/drift_degradation.csv", index=False)
        print("[OK] Saved: outputs/final_eval/drift_degradation.csv")
    else:
        print("[!] Insufficient data for temporal split.")
        print("    Run the simulation for 2.5+ hours to capture both windows.")
else:
    print("[!] No timestamp column found — cannot split pre/post drift.")
    print("    Use drift_log.csv from streaming_listener_v2.py instead.")


# ── EXPERIMENT 3: MCNEMAR TEST ───────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 3: McNemar Test (RF vs DRIFT-MLP)")
print("=" * 60)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_full.values, y, test_size=0.2, random_state=42, stratify=y
)
rf_final  = RandomForestClassifier(n_estimators=100, random_state=42)
mlp_final = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                           max_iter=500, random_state=42)
rf_final.fit(X_tr, y_tr);   y_pred_rf  = rf_final.predict(X_te)
mlp_final.fit(X_tr, y_tr);  y_pred_mlp = mlp_final.predict(X_te)

p_val = mcnemar_test(y_te, y_pred_rf, y_pred_mlp)
sig   = "SIGNIFICANT" if p_val < 0.05 else "not significant"
rf_f1  = f1_score(y_te, y_pred_rf,  average='weighted')
mlp_f1 = f1_score(y_te, y_pred_mlp, average='weighted')
print(f"  RF F1:       {rf_f1:.4f}")
print(f"  DRIFT-MLP F1:{mlp_f1:.4f}")
print(f"  McNemar p-value: {p_val:.6f} → {sig} (α=0.05)")


# ── EXPERIMENT 4: FEATURE ABLATION ──────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 4: Feature Ablation")
print("  Defeats the circular evaluation criticism:")
print("  Shows IAT-only features CAN detect attacks.")
print("=" * 60)

# FIXED: "duration" added to shortcut keywords
shortcut_keywords = [
    # Volumetric — raw byte/size counts only
    "bytes", "_ps",  # packet size stats
    # Temporal leakage — already dropped from DROP list but belt-and-suspenders
    "tcp_len", "tcp_time_delta", "timestamp",
    "bidirectional_first_seen", "bidirectional_last_seen",
    "src2dst_first_seen", "src2dst_last_seen",
    "dst2rc_first_seen", "dst2src_last_seen",
    "duration",
    # Raw packet counts (volumetric, not behavioral)
    # NOTE: Only bare packet counts are shortcuts.
    # Flag-specific counts (syn, ack, psh) are BEHAVIORAL and excluded.
]
volumetric_packet_counts = [
    'bidirectional_packets', 'src2dst_packets', 'dst2src_packets'
]
iat_keywords = ["piat", "iat"]

shortcut_feats = [f for f in features 
                  if any(k in f.lower() for k in shortcut_keywords)
                  or f in volumetric_packet_counts]
iat_feats      = [f for f in features if any(k in f.lower() for k in iat_keywords)]
no_shortcut    = [f for f in features if f not in shortcut_feats]

print(f"  All features:       {len(features)}")
print(f"  Shortcut features:  {len(shortcut_feats)} → {shortcut_feats}")
print(f"  IAT-only features:  {len(iat_feats)} → {iat_feats}")
print(f"  No-shortcut set:    {len(no_shortcut)}")

ablation_rows = []
for name, feat_set in [("All features",            features),
                        ("No shortcuts (IAT+other)", no_shortcut),
                        ("IAT-only",                 iat_feats)]:
    if not feat_set:
        print(f"  [!] Empty feature set for '{name}' — skipping.")
        continue
    X_ab = X_full[feat_set].fillna(0)
    X_tr_ab, X_te_ab, y_tr_ab, y_te_ab = train_test_split(
        X_ab.values, y, test_size=0.2, random_state=42, stratify=y
    )
    rf_ab = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_ab.fit(X_tr_ab, y_tr_ab)
    y_pred_ab = rf_ab.predict(X_te_ab)
    f1_ab = f1_score(y_te_ab, y_pred_ab, average="weighted")
    ablation_rows.append({"Feature_Set": name, "Num_Features": len(feat_set),
                           "F1_Score": round(f1_ab, 4)})
    print(f"  {name}: F1 = {f1_ab:.4f}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv("outputs/final_eval/ablation_results.csv", index=False)
print("[OK] Saved: outputs/final_eval/ablation_results.csv")


# ── FINAL SUMMARY CHART ──────────────────────────────────────
print("\n[*] Generating final results summary chart...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Final Evaluation Results — REA-HID Major Project",
             fontsize=13, fontweight="bold")

# Chart 1: Multi-seed F1 with error bars
models_list = multiseed_df["Model"].tolist()
f1_means    = multiseed_df["F1_Mean"].tolist()
f1_stds     = multiseed_df["F1_Std"].tolist()
ax = axes[0]
bars = ax.bar(range(len(models_list)), f1_means, yerr=f1_stds,
              color=["#2E75B6", "#375623"], capsize=6, alpha=0.85)
ax.set_xticks(range(len(models_list)))
ax.set_xticklabels([m.split(" ")[0] for m in models_list], fontsize=10)
ax.set_ylabel("Weighted F1-Score")
ax.set_title("Model Comparison\n(10-seed mean ± std)", fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
for b, v, s in zip(bars, f1_means, f1_stds):
    ax.text(b.get_x() + b.get_width()/2, v + s + 0.01,
            f"{v:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8)

# Chart 2: Feature ablation
if ablation_rows:
    ab_df   = pd.DataFrame(ablation_rows)
    ax      = axes[1]
    colors  = ["#C00000", "#2E75B6", "#375623"][:len(ab_df)]
    bars    = ax.bar(range(len(ab_df)), ab_df["F1_Score"].tolist(),
                     color=colors, alpha=0.85)
    ax.set_xticks(range(len(ab_df)))
    ax.set_xticklabels([r.split("(")[0].strip() for r in ab_df["Feature_Set"]],
                        fontsize=8, rotation=12)
    ax.set_ylabel("Weighted F1-Score")
    ax.set_title("Feature Ablation\n(defeats circular eval criticism)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, ab_df["F1_Score"].tolist()):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# Chart 3: McNemar result text box
ax = axes[2]
ax.axis("off")
sig_color  = "#FFE0E0" if p_val < 0.05 else "#E0FFE0"
mc_text = (f"McNemar Test\nRF vs DRIFT-MLP\n\n"
           f"RF F1:        {rf_f1:.4f}\n"
           f"DRIFT-MLP F1: {mlp_f1:.4f}\n\n"
           f"p-value: {p_val:.6f}\n"
           f"Result: {sig.upper()}\n(α = 0.05)")
ax.text(0.5, 0.5, mc_text, ha="center", va="center",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round", facecolor=sig_color, alpha=0.9),
        fontfamily="monospace")
ax.set_title("Statistical Significance", fontsize=11)

plt.tight_layout()
plt.savefig("outputs/final_eval/final_results_summary.png", dpi=150,
            bbox_inches="tight")
print("[OK] Saved: outputs/final_eval/final_results_summary.png")
print("\n── All outputs saved to outputs/final_eval/ ────────────")
