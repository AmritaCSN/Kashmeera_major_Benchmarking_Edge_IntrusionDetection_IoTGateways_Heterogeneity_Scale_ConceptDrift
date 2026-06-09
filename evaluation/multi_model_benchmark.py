"""
multi_model_benchmark.py
========================
Evaluates multiple ML and DL architectures across public IoT datasets
AND the custom REA-HID dataset.
Measures: F1-Score, False Positive Rate (FPR), and Inference Latency.
"""

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Safe for headless servers
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample

warnings.filterwarnings("ignore")
os.makedirs("outputs/model_benchmarks", exist_ok=True)

# ── PATH CONFIGURATION ──────────────────────────────────────────
MQTTEEB_D_PATH   = "../data/raw/MQTTEEB-D_Final_Dataset/Preprocessed_Data/MQTTEEB-D_cleaned_data.csv"
MQTT_IDS2020_DIR = "../data/raw/mqtt-iot-ids2020"
REA_HID_PATH     = "dataset_mqtt_coap_final.csv"
# ────────────────────────────────────────────────────────────────

ALWAYS_DROP_KEYWORDS = ["ip_src", "ip_dst", "src_ip", "dst_ip", "src_mac", "dst_mac", "src_port", "dst_port", "flow_id", "row_number", " id"]

def _drop_leaky_cols(df):
    to_drop = [c for c in df.columns if any(kw in c.lower() for kw in ALWAYS_DROP_KEYWORDS)]
    return df.drop(columns=to_drop, errors="ignore")

def _auto_detect_label(df):
    candidates = ["label", "Label", "class", "Class", "attack_type", "Attack_type", "target", "is_attack"]
    for c in candidates:
        if c in df.columns: return c
    return df.columns[-1]

def _encode_binary(series):
    s = series.astype(str).str.strip().str.lower()
    benign_vals = {"benign", "normal", "0", "false", "0.0", "legitimate", "benigntraffic"}
    return (~s.isin(benign_vals)).astype(int).values

def load_mqtteeb_d(path):
    if not os.path.exists(path): return None
    print(f"[*] Loading MQTTEEB-D from {path}...")
    df = pd.read_csv(path, low_memory=False)
    label_col = _auto_detect_label(df)
    df = df.dropna(subset=[label_col])
    attack_types = df[label_col].astype(str)
    df = _drop_leaky_cols(df).drop(columns=[label_col], errors="ignore")
    X = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    return X, _encode_binary(attack_types), "MQTTEEB-D (Public)"

def load_mqtt_ids2020(folder):
    if not os.path.exists(folder): return None
    biflow_files = [f for f in glob.glob(os.path.join(folder, "*biflow*.csv")) if "uniflow" not in f.lower()]
    if not biflow_files: return None
    print(f"[*] Loading MQTT-IoT-IDS2020 from {folder}...")
    frames = [pd.read_csv(fp, low_memory=False) for fp in sorted(biflow_files)]
    df = pd.concat(frames, ignore_index=True)
    if "is_attack" not in df.columns: return None
    y = df["is_attack"].astype(int).values
    df = _drop_leaky_cols(df).drop(columns=["is_attack", "proto", "_attack_type"], errors="ignore")
    X = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    return X, y, "MQTT-IDS2020 (Public)"

def load_rea_hid(path):
    if not os.path.exists(path): return None
    print(f"[*] Loading REA-HID from {path}...")
    df = pd.read_csv(path, low_memory=False)
    y = df['label'].astype(int).values
    
    # Strictly drop identifiers and temporal leakages (as proven in your earlier scripts)
    drop_cols = [
        'label', 'attack_type', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
        'src_oui', 'dst_oui', 'application_name', 'application_category_name',
        'id', 'src_port', 'dst_port',
        'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
        'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
        'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
        'bidirectional_duration_ms', 'src2dst_duration_ms', 'dst2src_duration_ms'
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    return X, y, "REA-HID (Ours)"

# ── MODEL CONFIGURATION ─────────────────────────────────────────
MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric='logloss'),
    "Deep Learning (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=150, random_state=42, early_stopping=True)
}
# ────────────────────────────────────────────────────────────────

def run_evaluation(X, y, dataset_name):
    print(f"\n[{dataset_name}] Evaluating Models...")
    
    benign_n, attack_n = int((y == 0).sum()), int((y == 1).sum())
    if max(benign_n, attack_n) / (min(benign_n, attack_n) + 1) > 5:
        target_n = min(min(benign_n, attack_n) * 3, max(benign_n, attack_n))
        maj_mask = (y == (0 if benign_n > attack_n else 1))
        X_maj, y_maj = X.values[maj_mask], y[maj_mask]
        X_min, y_min = X.values[~maj_mask], y[~maj_mask]
        X_maj_ds, y_maj_ds = resample(X_maj, y_maj, n_samples=target_n, random_state=42)
        X_bal, y_bal = np.vstack([X_maj_ds, X_min]), np.concatenate([y_maj_ds, y_min])
    else:
        X_bal, y_bal = X.values, y
        
    X_tr, X_te, y_tr, y_te = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    results = []
    
    for model_name, model in MODELS.items():
        print(f"  -> Training {model_name}...")
        
        t0 = time.time()
        model.fit(X_tr_scaled, y_tr)
        train_time = time.time() - t0
        
        t1 = time.time()
        y_pred = model.predict(X_te_scaled)
        inf_time_total = time.time() - t1
        inf_latency_ms = (inf_time_total / len(y_te)) * 1000 
        
        f1 = f1_score(y_te, y_pred, average="weighted")
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "F1_Weighted": f1,
            "FPR": fpr,
            "Train_Time_s": train_time,
            "Latency_ms_per_flow": inf_latency_ms
        })
        
    return results

if __name__ == "__main__":
    print("======================================================")
    print("Multi-Architecture Benchmarking (ML vs DL) on Edge")
    print("======================================================")
    
    datasets = [
        load_mqtteeb_d(MQTTEEB_D_PATH),
        load_mqtt_ids2020(MQTT_IDS2020_DIR),
        load_rea_hid(REA_HID_PATH)  # <--- YOUR DATASET IS NOW HERE
    ]
    
    all_results = []
    dataset_names = []
    
    for ds in datasets:
        if ds is not None:
            X, y, name = ds
            dataset_names.append(name)
            res = run_evaluation(X, y, name)
            all_results.extend(res)
            
    if not all_results:
        print("\n[!] No datasets found. Check paths at the top of the script.")
        exit(1)

    df_res = pd.DataFrame(all_results)
    df_res.to_csv("outputs/model_benchmarks/multi_model_results.csv", index=False)
    
    print("\n======================================================")
    print("FINAL RESULTS TABLE")
    print("======================================================")
    print(df_res.to_string(index=False))
    
    # ── GENERATE GROUPED BAR CHART ─────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(dataset_names))
    width = 0.25
    
    models = list(MODELS.keys())
    colors = ['#2E75B6', '#D46B08', '#2ca02c']
    
    for i, model in enumerate(models):
        f1_scores = df_res[df_res["Model"] == model]["F1_Weighted"].values
        offset = width * i
        bars = ax.bar(x + offset, f1_scores, width, label=model, color=colors[i], edgecolor='black')
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Weighted F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Detection Efficacy Across ML/DL Architectures (Public vs. Ours)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(dataset_names, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("outputs/model_benchmarks/multi_model_f1_comparison.png", dpi=300)
    print("\n[*] Graph saved to: outputs/model_benchmarks/multi_model_f1_comparison.png")