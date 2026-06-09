"""
shortcut_audit_multidataset.py  (v2 — major project 3rd review)
================================================================
Proves that shortcut learning is a UNIVERSAL problem across public IoT IDS
benchmarks — not an artefact of a single dataset.

Runs a complete audit on up to 4 datasets:
  1. MQTTEEB-D          (packet-level MQTT features, your primary dataset)
  2. mqtt-iot-ids2020   (Hindy et al. 2021 — multi-file, biflow schema)
  3. TON-IoT            (UNSW — if downloaded)
  4. CIC-IoT-2023       (UNB  — if downloaded)

For EACH dataset this script produces:
  A. Top-15 feature importance chart (Gini, colour-coded by shortcut type)
  B. SHAP summary plot (beeswarm) — stronger than Gini for reviewer
  C. Class balance bar chart
  D. Per-class attack-type breakdown (where label has multiple classes)
  E. Permutation importance (top 10) — proves Gini wasn't misleading
  F. Shortcut-Dominance Score (SDS): fraction of top-5 features that are
     volumetric or temporal shortcuts → one number per dataset
  G. Cross-dataset feature name mapping table (for dissertation Table X)

Final outputs:
  outputs/shortcut_audit/<DATASET>_gini_importance.png
  outputs/shortcut_audit/<DATASET>_shap_summary.png
  outputs/shortcut_audit/<DATASET>_permutation_importance.png
  outputs/shortcut_audit/<DATASET>_class_balance.png
  outputs/shortcut_audit/<DATASET>_feature_importance.csv
  outputs/shortcut_audit/<DATASET>_shap_values.csv
  outputs/shortcut_audit/shortcut_audit_combined.png   ← the main paper figure
  outputs/shortcut_audit/audit_summary.csv             ← Table for dissertation
  outputs/shortcut_audit/cross_dataset_feature_map.csv ← cross-domain table

DATASET PATHS — edit only these four lines if your paths differ:
"""

# ── PATH CONFIGURATION ───────────────────────────────────────────
# Edit these to match your server's actual paths.
MQTTEEB_D_PATH  = "../data/raw/MQTTEEB-D_Final_Dataset/Preprocessed_Data/MQTTEEB-D_cleaned_data.csv"
MQTT_IDS2020_DIR = "../data/raw/mqtt-iot-ids2020"          # folder containing the CSVs
TON_IOT_PATH    = "../data/raw/TON_IoT/ton_iot_network.csv"
CIC_IOT_PATH    = "../data/raw/CIC_IoT_2023/cic_iot_2023.csv"
REA_HID_PATH    = "dataset_mqtt_coap_final.csv"
# ────────────────────────────────────────────────────────────────

import os
import glob
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless — safe for IoT server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# optional — SHAP; gracefully skipped if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[!] shap not installed. Run: pip install shap --break-system-packages")
    print("    SHAP plots will be skipped but all other analysis runs fine.\n")

os.makedirs("outputs/shortcut_audit", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# 1.  SHORTCUT TAXONOMY
#     This is the central classification used throughout.
#     Reviewer-critical: these categories must be JUSTIFIED.
#
#     VOLUMETRIC: raw counts / sizes that an attacker can trivially
#       mimic (as your volumetric camouflage engine proves).
#     TEMPORAL_LEAKAGE: absolute timestamps or flow-duration fields
#       derived from first_seen / last_seen. These encode WHEN a flow
#       happened, not HOW it behaved. A model that uses these is
#       essentially memorising time-slots, not learning behaviour.
#     LEGITIMATE_IAT: statistical summaries of inter-arrival gaps
#       (mean, std, min, max IAT). These ARE genuine behavioural
#       features. Critically, fwd_mean_iat in mqtt-iot-ids2020 is
#       NOT a shortcut — it is the detection signal.
#       Do NOT colour IAT stats as shortcuts.
# ════════════════════════════════════════════════════════════════

VOLUMETRIC_KEYWORDS = [
    "tcp_len", "pkt_len", "packet_length", "packet_size",
    "fwd_num_bytes", "bwd_num_bytes", "num_bytes",
    "bidirectional_bytes", "src2dst_bytes", "dst2src_bytes",
    "total_fwd_packet", "total_bwd_packet",
    "fwd_num_pkts", "bwd_num_pkts",
    "fwd_packet_length_max", "bwd_packet_length_max",
    "avg_fwd_segment_size", "avg_bwd_segment_size",
    "fwd_header_len", "bwd_header_len",
    "total_length_of_fwd_packet", "total_length_of_bwd_packet",
    "fwd_seg_size_min", "fwd_act_data_pkt","src_pkts", "dst_pkts", "src_bytes", "dst_bytes",
    "num_pkts", "pkts",
    # CICFlowMeter names
    "totlen_fwd_pkts", "totlen_bwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_min",
    "bwd_pkt_len_max", "bwd_pkt_len_min",
    "pkt_len_max", "pkt_len_min", "pkt_len_mean", "pkt_len_std",
    "pkt_size_avg",
]

TEMPORAL_LEAKAGE_KEYWORDS = [
    # Absolute timestamp fields — direct leakage
    "timestamp", "tcp_time_delta",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    # Duration = last_seen - first_seen = indirect timestamp leakage
    "bidirectional_duration", "src2dst_duration", "dst2src_duration",
    "flow_duration",
    # CICFlowMeter flow duration
    "flow_duration",
    # Total IAT (sum of gaps) encodes absolute elapsed time
    "fwd_iat_total", "bwd_iat_total", "flow_iat_total",
    # Active / idle time — encode session timing
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
]

# NOTE: fwd_mean_iat, bwd_mean_iat, fwd_std_iat, bwd_std_iat,
# fwd_min_iat, bwd_max_iat etc. are LEGITIMATE BEHAVIOURAL features.
# flow_iat_mean, flow_iat_std, flow_iat_min, flow_iat_max are also
# legitimate. Only the TOTAL (sum) is a leakage proxy.

ALWAYS_DROP_KEYWORDS = [
    "ip_src", "ip_dst", "src_ip", "dst_ip",
    "src_mac", "dst_mac", "src_oui", "dst_oui",
    "prt_src", "prt_dst", "src_port", "dst_port",
    "flow_id", "row_number", " id",
]


def classify_feature(name: str) -> str:
    """
    Classify a feature name into one of three categories.
    Returns: 'volumetric' | 'temporal_leak' | 'legitimate'
    """
    n = name.lower().strip()
    # Check temporal leakage first (higher priority — duration is leakage
    # even though it sounds behavioural)
    for kw in TEMPORAL_LEAKAGE_KEYWORDS:
        if kw in n:
            return "temporal_leak"
    for kw in VOLUMETRIC_KEYWORDS:
        if kw in n:
            return "volumetric"
    return "legitimate"


def feature_color(name: str) -> str:
    cat = classify_feature(name)
    return {"volumetric": "#C00000",
            "temporal_leak": "#D46B08",
            "legitimate": "#2E75B6"}[cat]


def shortcut_dominance_score(top_features: list, k: int = 5) -> float:
    """
    SDS = fraction of top-k features that are shortcuts.
    SDS = 1.0 means model relies ENTIRELY on non-behavioural features.
    SDS = 0.0 means model learned purely from behavioural signals.
    """
    top_k = top_features[:k]
    shortcuts = sum(1 for f in top_k
                    if classify_feature(f) in ("volumetric", "temporal_leak"))
    return round(shortcuts / k, 2)


# ════════════════════════════════════════════════════════════════
# 2.  DATASET LOADERS
#     Each loader returns (X, y, attack_type_series, dataset_info_dict)
#     X: numeric features only, leaky columns dropped
#     y: binary numpy array (0=benign, 1=attack)
#     attack_type_series: string labels for per-class breakdown
#     dataset_info_dict: metadata for the summary table
# ════════════════════════════════════════════════════════════════

def _drop_leaky_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns
               if any(kw in c.lower() for kw in ALWAYS_DROP_KEYWORDS)]
    return df.drop(columns=to_drop, errors="ignore")


def _auto_detect_label(df: pd.DataFrame) -> str:
    candidates = ["label", "Label", "class", "Class", "attack_type",
                  "Attack_type", "attack", "category", "type",
                  "target", "is_attack", "Is_attack"]
    for c in candidates:
        if c in df.columns:
            return c
    # last resort: column whose unique values include 'benign'/'normal'
    for c in df.columns:
        vals = df[c].astype(str).str.lower().unique()
        if any(v in vals for v in ("benign", "normal", "attack")):
            return c
    raise ValueError(
        f"Cannot auto-detect label column. "
        f"Columns: {list(df.columns)[:15]}"
    )


def _encode_binary(series: pd.Series) -> np.ndarray:
    """Encode any label column to binary 0/1."""
    s = series.astype(str).str.strip().str.lower()
    # Added "legitimate" for MQTTEEB-D compatibility
    benign_vals = {"benign", "normal", "0", "false", "0.0", "legitimate","benigntraffic"}
    return (~s.isin(benign_vals)).astype(int).values


def load_mqtteeb_d(path: str):
    """
    MQTTEEB-D loader.
    Schema: packet-level MQTT flows.
    Label column: auto-detected.
    """
    if not os.path.exists(path):
        return None
    print(f"  Loading MQTTEEB-D from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    label_col = _auto_detect_label(df)
    df = df.dropna(subset=[label_col])
    attack_types = df[label_col].astype(str)

    df = _drop_leaky_cols(df)
    df = df.drop(columns=[label_col], errors="ignore")
    X = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    y = _encode_binary(attack_types)

    info = {
        "full_name": "MQTTEEB-D (Chaudhari et al. 2023)",
        "protocol": "MQTT",
        "feature_schema": "Packet-level (Wireshark/tshark dissection)",
        "n_rows": len(df),
        "label_col": label_col,
    }
    return X, y, attack_types, info


def load_mqtt_ids2020(folder: str):
    """
    mqtt-iot-ids2020 (Hindy et al. 2021) loader.

    IMPORTANT DESIGN DECISIONS (justify in dissertation):
    1. Only biflow CSVs are loaded — uniflow_normal is EXCLUDED because
       it lacks bwd_ features, making schema incompatible.
    2. All biflow files are concatenated: normal + scan_a + scan_su +
       sparta + bruteforce. The is_attack column (0/1) is the label.
    3. ip_src, ip_dst, prt_src, prt_dst are dropped (identifier leakage).
    4. fwd/bwd IAT statistical features (mean, std, min, max) are
       classified as LEGITIMATE — they are the true behavioural signal.
    5. fwd_num_bytes / bwd_num_bytes are classified as VOLUMETRIC shortcuts.
    """
    if not os.path.exists(folder):
        return None

    # Load only biflow CSVs — explicitly exclude uniflow
    biflow_pattern = os.path.join(folder, "*.csv")
    biflow_files = [f for f in glob.glob(biflow_pattern)
                    if "uniflow" not in f.lower()
                    and "merged" not in f.lower()]

    if not biflow_files:
        print(f"  [!] No biflow CSVs found in {folder}")
        return None

    print(f"  Loading mqtt-iot-ids2020 from {len(biflow_files)} biflow files ...")
    frames = []
    for fp in sorted(biflow_files):
        fname = os.path.basename(fp)
        chunk = pd.read_csv(fp, low_memory=False)
        # Derive attack_type label from filename for per-class breakdown
        if "normal" in fname.lower():
            chunk["_attack_type"] = "Benign"
        elif "scana" in fname.lower() and "scansu" not in fname.lower():
            chunk["_attack_type"] = "Scan_A"
        elif "scansu" in fname.lower():
            chunk["_attack_type"] = "Scan_SU"
        elif "sparta" in fname.lower():
            chunk["_attack_type"] = "Sparta"
        elif "bruteforce" in fname.lower():
            chunk["_attack_type"] = "BruteForce"
        else:
            chunk["_attack_type"] = "Unknown"
        frames.append(chunk)
        print(f"    {fname}: {len(chunk)} rows → {chunk['_attack_type'].iloc[0]}")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Merged shape: {df.shape}")

    # Verify is_attack column exists
    if "is_attack" not in df.columns:
        print("  [!] 'is_attack' column not found after merge. Check filenames.")
        return None

    attack_types = df["_attack_type"].astype(str)
    y = df["is_attack"].astype(int).values

    df = _drop_leaky_cols(df)
    drop_extra = ["is_attack", "_attack_type", "proto"]  # proto is categorical
    df = df.drop(columns=drop_extra, errors="ignore")
    X = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)

    info = {
        "full_name": "MQTT-IoT-IDS2020 (Hindy et al. 2021)",
        "protocol": "MQTT (TCP port 1883)",
        "feature_schema": "Biflow — fwd/bwd IAT stats + pkt length stats + TCP flags",
        "n_rows": len(df),
        "label_col": "is_attack",
    }
    return X, y, attack_types, info


def load_ton_iot(path: str):
    """
    TON-IoT network dataset (Moustafa et al. 2021) loader.
    Label column: 'type' (multi-class: normal, ddos, dos, injection, etc.)
    """
    if not os.path.exists(path):
        return None
    print(f"  Loading TON-IoT from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # TON-IoT uses 'type' for attack type and 'label' for binary (0/1)
    if "type" in df.columns:
        attack_types = df["type"].astype(str)
        label_col = "label" if "label" in df.columns else "type"
    else:
        label_col = _auto_detect_label(df)
        attack_types = df[label_col].astype(str)

    df = df.dropna(subset=[label_col])
    attack_types = attack_types.loc[df.index]

    df = _drop_leaky_cols(df)
    # Drop both label columns to avoid leakage
    df = df.drop(columns=["label", "type"], errors="ignore")
    X = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    y = _encode_binary(attack_types)

    info = {
        "full_name": "TON-IoT (Moustafa et al. 2021)",
        "protocol": "Multi-protocol (TCP/UDP/ICMP)",
        "feature_schema": "CICFlowMeter-style network flows",
        "n_rows": len(df),
        "label_col": label_col,
    }
    return X, y, attack_types, info


def load_cic_iot_2023(path: str):
    """
    CIC-IoT-2023 (Neto et al. 2023) loader.
    Label column: auto-detected.
    """
    if not os.path.exists(path):
        return None
    print(f"  Loading CIC-IoT-2023 from {path} ...")
    # This dataset can be large — sample if needed
    df = pd.read_csv(path, low_memory=False, nrows=300_000)
    print(f"  Raw shape (capped at 300K rows): {df.shape}")

    label_col = _auto_detect_label(df)
    df = df.dropna(subset=[label_col])
    attack_types = df[label_col].astype(str)

    df = _drop_leaky_cols(df)
    df = df.drop(columns=[label_col], errors="ignore")
    X = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    y = _encode_binary(attack_types)

    info = {
        "full_name": "CIC-IoT-2023 (Neto et al. 2023)",
        "protocol": "Multi-protocol IoT",
        "feature_schema": "CICFlowMeter-style network flows",
        "n_rows": len(df),
        "label_col": label_col,
    }
    return X, y, attack_types, info


def load_rea_hid(path: str):
    """
    REA-HID (Your Custom Dataset) loader.
    """
    if not os.path.exists(path):
        return None
    print(f"  Loading REA-HID from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    
    label_col = "label"
    attack_types = df["attack_type"].astype(str)

    # Drop identifiers and temporal leakages specific to REA-HID
    drop_cols = [
        'label', 'attack_type', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
        'src_oui', 'dst_oui', 'application_name', 'application_category_name',
        'id', 'src_port', 'dst_port',
        'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
        'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
        'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
        'bidirectional_duration_ms', 'src2dst_duration_ms', 'dst2src_duration_ms',
    ]
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    y = df[label_col].astype(int).values

    info = {
        "full_name": "REA-HID (Proposed Dataset)",
        "protocol": "MQTT + CoAP",
        "feature_schema": "NFStream Flow-level",
        "n_rows": len(df),
        "label_col": label_col,
    }
    return X, y, attack_types, info


# ════════════════════════════════════════════════════════════════
# 3.  CORE AUDIT FUNCTION
#     Runs on one (X, y) pair and returns all results.
# ════════════════════════════════════════════════════════════════

def run_audit(X: pd.DataFrame, y: np.ndarray, name: str,
              attack_types: pd.Series, info: dict) -> dict:
    """
    Full audit pipeline for one dataset.
    Returns a result dict consumed by the plotting and summary functions.
    """
    print(f"\n  {'─'*55}")
    print(f"  AUDIT: {name}")
    print(f"  Samples: {len(X):,} | Features: {X.shape[1]} | "
          f"Benign: {(y==0).sum():,} | Attack: {(y==1).sum():,}")
    print(f"  Class ratio  B:A = "
          f"1:{(y==1).sum()/(y==0).sum():.2f}" if (y==0).sum() > 0 else "")

    # ── Class imbalance check ────────────────────────────────────
    # If imbalance > 10:1 in either direction, downsample majority class
    # to 5x minority to keep RF from being dominated by majority.
    # We record the original counts for reporting.
    benign_n, attack_n = int((y == 0).sum()), int((y == 1).sum())
    if max(benign_n, attack_n) / (min(benign_n, attack_n) + 1) > 10:
        print(f"  [!] Severe imbalance detected. Downsampling majority class.")
        minority = min(benign_n, attack_n)
        target_n = min(minority * 5, max(benign_n, attack_n))
        X_arr = X.values
        maj_mask = (y == (0 if benign_n > attack_n else 1))
        X_maj, y_maj = X_arr[maj_mask], y[maj_mask]
        X_min, y_min = X_arr[~maj_mask], y[~maj_mask]
        X_maj_ds, y_maj_ds = resample(X_maj, y_maj,
                                       n_samples=target_n,
                                       random_state=42)
        X_bal = np.vstack([X_maj_ds, X_min])
        y_bal = np.concatenate([y_maj_ds, y_min])
        # Shuffle
        idx = np.random.RandomState(42).permutation(len(X_bal))
        X_bal, y_bal = X_bal[idx], y_bal[idx]
        X_work = pd.DataFrame(X_bal, columns=X.columns)
        y_work = y_bal
        print(f"  After balancing: {(y_work==0).sum():,} benign, "
              f"{(y_work==1).sum():,} attack")
    else:
        X_work, y_work = X, y

    # ── Train / test split ───────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_work, y_work, test_size=0.2, random_state=42, stratify=y_work
    )

    # ── Random Forest ────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",  # handles any residual imbalance
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    f1_w    = f1_score(y_te, y_pred, average="weighted")
    f1_mac  = f1_score(y_te, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"  RF  weighted-F1={f1_w:.4f} | macro-F1={f1_mac:.4f} | "
          f"AUC={auc:.4f} | FPR={fpr:.4f}")
    cr = classification_report(y_te, y_pred,
                                target_names=["Benign", "Attack"],
                                output_dict=True)

    # ── Gini feature importance (top 15) ────────────────────────
    gini_imp = pd.Series(rf.feature_importances_, index=X.columns)
    gini_imp = gini_imp.sort_values(ascending=False)
    top15_gini = gini_imp.head(15)

    # ── Permutation importance (top 10, 5 repeats) ──────────────
    # Permutation importance directly measures F1 drop when a feature
    # is randomly shuffled. Unlike Gini, it is not biased toward
    # high-cardinality features (Strobl et al. 2007).
    print(f"  Computing permutation importance (5 repeats, test set)...")
    perm = permutation_importance(
        rf, X_te, y_te,
        n_repeats=5,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1,
    )
    perm_imp = pd.Series(perm.importances_mean, index=X.columns)
    perm_std = pd.Series(perm.importances_std,  index=X.columns)
    perm_imp = perm_imp.sort_values(ascending=False)
    perm_std = perm_std.loc[perm_imp.index]
    top10_perm = perm_imp.head(10)

    # ── SDS computation ─────────────────────────────────────────
    sds = shortcut_dominance_score(list(top15_gini.index), k=5)
    print(f"  Shortcut Dominance Score (SDS, k=5): {sds:.2f}  "
          f"({'HIGH — model is cheating' if sds >= 0.6 else 'MODERATE' if sds >= 0.4 else 'LOW — legitimate features dominate'})")

    # ── SHAP (if available) ──────────────────────────────────────
    shap_vals = None
    shap_feat_imp = None
    if SHAP_AVAILABLE:
        print(f"  Computing SHAP values (TreeExplainer, subsample 1000)...")
        try:
            # Subsample test set for speed
            n_shap = min(1000, len(X_te))
            idx_shap = np.random.RandomState(42).choice(len(X_te) if isinstance(X_te, np.ndarray) else len(X_te),n_shap, replace=False)
            X_shap = X_te.iloc[idx_shap] if hasattr(X_te, 'iloc') else X_te[idx_shap]

            explainer  = shap.TreeExplainer(rf)
            shap_out   = explainer.shap_values(X_shap)
            
            # Handle SHAP output structures for binary classification
            if isinstance(shap_out, list) and len(shap_out) == 2:
                sv = shap_out[1]  # Older SHAP version (list of arrays)
            elif isinstance(shap_out, np.ndarray) and len(shap_out.shape) == 3:
                sv = shap_out[:, :, 1]  # Newer SHAP version (3D array)
            else:
                sv = shap_out

            shap_vals      = sv
            shap_feat_imp  = pd.Series(np.abs(sv).mean(axis=0),index=X.columns).sort_values(ascending=False)

            # Save SHAP values CSV
            shap_df = pd.DataFrame(sv, columns=X.columns)
            shap_df.to_csv(f"outputs/shortcut_audit/{name}_shap_values.csv",index=False)
        except Exception as e:
            print(f"  [!] SHAP failed: {e}")
            shap_vals = None

    # ── Save Gini importance CSV ─────────────────────────────────
    gini_imp_df = pd.DataFrame({
        "feature": gini_imp.index,
        "gini_importance": gini_imp.values,
        "shortcut_type": [classify_feature(f) for f in gini_imp.index],
    })
    gini_imp_df.to_csv(
        f"outputs/shortcut_audit/{name}_feature_importance.csv",
        index=False)

    return {
        "name":          name,
        "info":          info,
        "X":             X,
        "X_te":          X_te,
        "y_te":          y_te,
        "y_pred":        y_pred,
        "attack_types":  attack_types,
        "benign_n":      benign_n,
        "attack_n":      attack_n,
        "f1_weighted":   f1_w,
        "f1_macro":      f1_mac,
        "auc":           auc,
        "fpr":           fpr,
        "cm":            cm,
        "cr":            cr,
        "top15_gini":    top15_gini,
        "top10_perm":    top10_perm,
        "perm_std":      perm_std,
        "shap_vals":     shap_vals,
        "shap_feat_imp": shap_feat_imp,
        "sds":           sds,
    }


# ════════════════════════════════════════════════════════════════
# 4.  INDIVIDUAL FIGURE GENERATORS
# ════════════════════════════════════════════════════════════════

def _color_list(feature_names):
    return [feature_color(f) for f in feature_names]


def plot_gini(result: dict):
    name    = result["name"]
    top     = result["top15_gini"]
    f1      = result["f1_weighted"]
    sds     = result["sds"]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = _color_list(top.index)
    bars    = ax.barh(top.index[::-1], top.values[::-1],
                      color=colors[::-1], edgecolor="white", linewidth=0.4)

    # Annotate shortcut type on bar
    for bar, feat in zip(bars[::-1], top.index[::-1]):
        cat = classify_feature(feat)
        if cat != "legitimate":
            label = "VOL" if cat == "volumetric" else "TEMP"
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    label, va="center", fontsize=7, color="#888", style="italic")

    ax.set_title(
        f"{name} — Top-15 Feature Importance (Gini)\n"
        f"RF weighted-F1 = {f1:.3f}  |  SDS = {sds:.2f}  "
        f"({'SHORTCUT-DOMINATED' if sds >= 0.6 else 'MIXED' if sds >= 0.4 else 'BEHAVIOUR-DRIVEN'})",
        fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Legend
    patches = [
        mpatches.Patch(color="#C00000", label="Volumetric shortcut (packet size / byte count)"),
        mpatches.Patch(color="#D46B08", label="Temporal leakage (timestamp / duration)"),
        mpatches.Patch(color="#2E75B6", label="Legitimate behavioural feature (IAT, flags)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    out = f"outputs/shortcut_audit/{name}_gini_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_permutation(result: dict):
    name = result["name"]
    top  = result["top10_perm"]
    std  = result["perm_std"].loc[top.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = _color_list(top.index)
    ax.barh(top.index[::-1], top.values[::-1],
            xerr=std.values[::-1],
            color=colors[::-1],
            edgecolor="white", linewidth=0.4,
            capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#555"})

    ax.set_title(
        f"{name} — Permutation Importance (F1 drop, 5 repeats)\n"
        "More reliable than Gini — directly measures predictive contribution",
        fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Mean F1 Decrease ± std (higher = more important)", fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    patches = [
        mpatches.Patch(color="#C00000", label="Volumetric shortcut"),
        mpatches.Patch(color="#D46B08", label="Temporal leakage"),
        mpatches.Patch(color="#2E75B6", label="Legitimate feature"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    out = f"outputs/shortcut_audit/{name}_permutation_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_shap(result: dict):
    if not SHAP_AVAILABLE or result["shap_vals"] is None:
        return
    name = result["name"]
    sv   = result["shap_vals"]
    X_sh = result["X_te"]
    n    = min(1000, len(X_sh))
    idx  = np.random.RandomState(42).choice(len(X_sh), n, replace=False)
    X_sub = X_sh.iloc[idx] if hasattr(X_sh, "iloc") else X_sh[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv[:n], X_sub,
        feature_names=list(result["X"].columns),
        plot_type="dot",
        max_display=15,
        show=False,
        color_bar=True,
    )
    plt.title(
        f"{name} — SHAP Beeswarm (Attack class)\n"
        "Each dot = one sample. X-axis = impact on attack prediction.",
        fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    out = f"outputs/shortcut_audit/{name}_shap_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_class_balance(result: dict):
    name         = result["name"]
    attack_types = result["attack_types"]

    vc = attack_types.value_counts()
    fig, ax = plt.subplots(figsize=(max(6, len(vc) * 1.2), 4))
    bar_colors = ["#2E75B6" if "benign" in v.lower() or "normal" in v.lower()
                  else "#C00000" for v in vc.index]
    ax.bar(vc.index, vc.values, color=bar_colors, edgecolor="white")
    ax.set_title(f"{name} — Class Distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of flows", fontsize=10)
    ax.set_xlabel("Traffic class", fontsize=10)
    for i, (v, c) in enumerate(zip(vc.index, vc.values)):
        ax.text(i, c + max(vc.values) * 0.01, f"{c:,}",
                ha="center", fontsize=8, rotation=30)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = f"outputs/shortcut_audit/{name}_class_balance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════
# 5.  COMBINED FIGURE (main paper figure — shortcut_audit_combined.png)
#     One panel per dataset showing Gini top-10 in colour.
#     Layout adapts to number of datasets (1–4).
# ════════════════════════════════════════════════════════════════

def plot_combined(results: list):
    n = len(results)
    if n == 0:
        return
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig_w = ncols * 10
    fig_h = nrows * 6

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for i, res in enumerate(results):
        ax   = axes_flat[i]
        top  = res["top15_gini"].head(10)
        cols = _color_list(top.index)
        ax.barh(top.index[::-1], top.values[::-1],
                color=cols[::-1], edgecolor="white", linewidth=0.3)

        sds_label = ("SHORTCUT-DOM." if res["sds"] >= 0.6
                     else "MIXED" if res["sds"] >= 0.4 else "LEGIT-DOM.")
        ax.set_title(
            f"{res['name']}\n"
            f"F1={res['f1_weighted']:.3f}  AUC={res['auc']:.3f}  "
            f"SDS={res['sds']:.2f} [{sds_label}]",
            fontsize=10, fontweight="bold")
        ax.set_xlabel("Gini Importance", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes_flat[j].set_visible(False)

    # Shared legend
    patches = [
        mpatches.Patch(color="#C00000",
                       label="Volumetric shortcut (byte/packet count)"),
        mpatches.Patch(color="#D46B08",
                       label="Temporal leakage (timestamp/duration)"),
        mpatches.Patch(color="#2E75B6",
                       label="Legitimate behavioural feature"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Shortcut Learning Audit: Top-10 Feature Importance Across IoT IDS Benchmarks\n"
        "Red/Orange = model exploiting non-behavioural shortcuts  "
        "|  SDS = Shortcut Dominance Score (0–1)",
        fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = "outputs/shortcut_audit/shortcut_audit_combined.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[*] Combined figure saved: {out}")


# ════════════════════════════════════════════════════════════════
# 6.  SUMMARY TABLE + CROSS-DATASET FEATURE MAP
# ════════════════════════════════════════════════════════════════

def save_summary(results: list):
    rows = []
    for res in results:
        top5 = list(res["top15_gini"].index[:5])
        rows.append({
            "Dataset":              res["name"],
            "Full_Name":            res["info"]["full_name"],
            "Protocol":             res["info"]["protocol"],
            "Feature_Schema":       res["info"]["feature_schema"],
            "N_Samples":            res["info"]["n_rows"],
            "N_Features":           res["X"].shape[1],
            "Benign_Count":         res["benign_n"],
            "Attack_Count":         res["attack_n"],
            "RF_F1_Weighted":       round(res["f1_weighted"], 4),
            "RF_F1_Macro":          round(res["f1_macro"],    4),
            "AUC_ROC":              round(res["auc"],         4),
            "FPR":                  round(res["fpr"],         4),
            "SDS_k5":               res["sds"],
            "Top1_Feature":         top5[0] if len(top5) > 0 else "",
            "Top1_Type":            classify_feature(top5[0]) if top5 else "",
            "Top2_Feature":         top5[1] if len(top5) > 1 else "",
            "Top3_Feature":         top5[2] if len(top5) > 2 else "",
            "Shortcut_Dominated":   res["sds"] >= 0.6,
        })
    df = pd.DataFrame(rows)
    df.to_csv("outputs/shortcut_audit/audit_summary.csv", index=False)

    print("\n" + "═"*70)
    print("AUDIT SUMMARY")
    print("═"*70)
    for _, row in df.iterrows():
        flag = "⚠ SHORTCUT" if row["Shortcut_Dominated"] else "✓ OK"
        print(f"  {row['Dataset']:<22} F1={row['RF_F1_Weighted']:.3f}  "
              f"AUC={row['AUC_ROC']:.3f}  SDS={row['SDS_k5']:.2f}  {flag}")
        print(f"    Top feature: {row['Top1_Feature']} ({row['Top1_Type']})")
    print(f"\n  Shortcut-dominated datasets: "
          f"{df[df['Shortcut_Dominated']]['Dataset'].tolist()}")
    print(f"  [CSV] outputs/shortcut_audit/audit_summary.csv")


def save_cross_dataset_feature_map(results: list):
    """
    Builds a table mapping conceptually equivalent features
    across datasets, showing how the same 'shortcut' manifests
    under different naming conventions.
    This is Table X in your dissertation — addresses the
    cross-domain generalizability critique (Reviewer 4, point C).
    """
    if len(results) < 2:
        return

    concept_map = {
        "Byte count (fwd)":   ["fwd_num_bytes",  "src2dst_bytes",   "totlen_fwd_pkts",  "total_fwd_bytes"],
        "Byte count (bwd)":   ["bwd_num_bytes",  "dst2src_bytes",   "totlen_bwd_pkts",  "total_bwd_bytes"],
        "Pkt count (fwd)":    ["fwd_num_pkts",   "src2dst_packets", "total_fwd_packets","totfwdpackets"],
        "Pkt count (bwd)":    ["bwd_num_pkts",   "dst2src_packets", "total_bwd_packets","totbwdpackets"],
        "Mean IAT (fwd)":     ["fwd_mean_iat",   "fwd_iat_mean",    "flow_iat_mean",    "bidirectional_mean_piat_ms"],
        "Std IAT (fwd)":      ["fwd_std_iat",    "fwd_iat_std",     "flow_iat_std",     "bidirectional_stddev_piat_ms"],
        "Flow duration":      ["flow_duration",  "bidirectional_duration_ms", "duration"],
        "Pkt length (mean)":  ["fwd_mean_pkt_len","fwd_pkt_len_mean","pkt_len_mean",    "bidirectional_mean_ps"],
        "TCP flags (PSH)":    ["fwd_num_psh_flags","psh_flag_cnt",   "fwd_psh_flags"],
        "TCP len / pkt size": ["tcp_len",         "pkt_size_avg",   "avg_fwd_segment_size"],
    }

    rows = []
    for concept, variants in concept_map.items():
        row = {"Semantic Concept": concept,
               "Shortcut?": classify_feature(variants[0])}
        for res in results:
            feat_set = set(c.lower() for c in res["X"].columns)
            found = next((v for v in variants if v.lower() in feat_set), "—")
            row[res["name"]] = found
        rows.append(row)

    map_df = pd.DataFrame(rows)
    map_df.to_csv(
        "outputs/shortcut_audit/cross_dataset_feature_map.csv",
        index=False)
    print(f"\n  [CSV] Cross-dataset feature map saved: "
          f"outputs/shortcut_audit/cross_dataset_feature_map.csv")

    print("\n── Cross-Dataset Feature Name Mapping ──────────────────")
    print(map_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════
# 7.  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("SHORTCUT AUDIT — Multi-Dataset IoT IDS Benchmarking")
    print("Major Project 3rd Review | Kashmeera R")
    print("=" * 70)

    # ── Load datasets ────────────────────────────────────────────
    # Adjust paths at top of file if needed.
    # Each loader returns None if file not found — gracefully skipped.

    raw_data = [
        ("MQTTEEB-D",       load_mqtteeb_d(MQTTEEB_D_PATH)),
        ("MQTT-IoT-IDS2020", load_mqtt_ids2020(MQTT_IDS2020_DIR)),
        ("TON-IoT",          load_ton_iot(TON_IOT_PATH)),
        ("CIC-IoT-2023",     load_cic_iot_2023(CIC_IOT_PATH)),
        ("REA-HID (Ours)",   load_rea_hid(REA_HID_PATH)),
    ]

    available = [(name, data) for name, data in raw_data if data is not None]

    if not available:
        print("\n[!] No datasets found. Check paths at top of script.")
        print("    Minimum required: MQTTEEB-D and/or MQTT-IoT-IDS2020")
        exit(1)

    print(f"\n[*] Datasets loaded: {[n for n, _ in available]}")

    # ── Run audit on each dataset ────────────────────────────────
    results = []
    for ds_name, (X, y, attack_types, info) in available:
        res = run_audit(X, y, ds_name, attack_types, info)
        results.append(res)
        # Individual plots
        plot_gini(res)
        plot_permutation(res)
        plot_shap(res)
        plot_class_balance(res)

    # ── Combined figure ──────────────────────────────────────────
    plot_combined(results)

    # ── Summary tables ───────────────────────────────────────────
    save_summary(results)
    save_cross_dataset_feature_map(results)

    # ── Final console summary ────────────────────────────────────
    print("\n" + "═" * 70)
    print("ALL OUTPUTS SAVED TO: outputs/shortcut_audit/")
    print("=" * 70)
    print("Files generated per dataset:")
    print("  <name>_gini_importance.png      ← Gini bar chart (colour-coded)")
    print("  <name>_permutation_importance.png ← Permutation importance ± std")
    print("  <name>_shap_summary.png         ← SHAP beeswarm (if shap installed)")
    print("  <name>_class_balance.png        ← Class distribution chart")
    print("  <name>_feature_importance.csv   ← Gini + shortcut_type per feature")
    print("  <name>_shap_values.csv          ← Raw SHAP values (1000 samples)")
    print("Shared:")
    print("  shortcut_audit_combined.png     ← MAIN PAPER FIGURE")
    print("  audit_summary.csv               ← Dissertation summary table")
    print("  cross_dataset_feature_map.csv   ← Cross-domain feature name mapping")
    print("=" * 70)

    if not SHAP_AVAILABLE:
        print("\n[REMINDER] Install SHAP for beeswarm plots:")
        print("  pip install shap --break-system-packages")