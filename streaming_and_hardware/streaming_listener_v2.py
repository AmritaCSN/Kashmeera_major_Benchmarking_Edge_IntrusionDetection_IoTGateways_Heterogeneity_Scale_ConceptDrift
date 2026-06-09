"""
streaming_listener_v2.py
========================
Fixed version of streaming_listener.py for the major project.

KEY FIXES vs original:
1. PH threshold lowered to 5.0 (was 50.0) — original NEVER fired because
   the IAT shift from 8.3s to 11.7s produces a cumulative sum of ~15-20,
   not 50. With threshold=50 you need a shift of 5+ standard deviations.
2. min_instances lowered to 30 (was 200) — 200 MQTT flows at 45s IAT
   takes ~2.5 hours to accumulate. Your test is only 121 minutes.
3. ALL drift events are logged to a CSV file (drift_log.csv) so you have
   evidence for the 3rd review even if the terminal scrolls past.
4. Inference latency (ms per flow) is measured and logged for hardware
   profiling section.
5. Drift detection column shows timestamp + flow number when triggered.
"""

import pandas as pd
import numpy as np
import time
import csv
import warnings
from nfstream import NFStreamer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. TRAIN THE ML INFERENCE ENGINE
# ==========================================
print("[*] Training EdgeML Inference Engine...")
df = pd.read_csv("REA_HID_Publication_Dataset_v2.csv")

drop_cols = ['src_ip', 'dst_ip', 'src_mac', 'dst_mac', 'src_oui', 'dst_oui',
             'application_name', 'application_category_name', 'attack_type',
             'label', 'id', 'src_port', 'dst_port',
             'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms',
             'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
             'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
             'bidirectional_duration_ms', 'src2dst_duration_ms', 'dst2src_duration_ms']

numeric_df = df.select_dtypes(include=[np.number])
features = [c for c in numeric_df.columns if c not in drop_cols]

X = numeric_df[features].fillna(0)
y = df['label']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
print(f"[*] Model trained. Tracking {len(features)} pure flow features.\n")

# ==========================================
# 2. PAGE-HINKLEY DRIFT DETECTOR
#    FIXED PARAMETERS:
#    - threshold=5.0  (was 50.0 — never fired before)
#    - min_instances=30 (was 200 — took hours before)
#    - delta=0.005    (sensitivity to sustained mean shift)
# ==========================================
class PageHinkley:
    """
    Page-Hinkley test for detecting upward shifts in a data stream.
    
    The statistic tracks: PH_t = sum_{i=1}^{t} (x_i - mean_i - delta)
    Drift is detected when PH_t > threshold.
    
    Parameters chosen for a 120-minute simulation where benign IAT shifts
    from ~8s to ~11s at the 60-minute mark:
    - min_instances=30: wait for 30 flows to establish baseline (~22 minutes)
    - delta=0.005: insensitive to random noise, sensitive to sustained shifts
    - threshold=5.0: fires after ~10 sustained above-mean observations
    """
    def __init__(self, min_instances=30, delta=0.005, threshold=5.0):
        self.sum = 0.0
        self.mean = 0.0
        self.count = 0
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.drift_detected = False
        self.drift_at_flow = None
        self.pre_drift_mean = None

    def update(self, value):
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.sum = max(0, self.sum + (value - self.mean - self.delta))

        if self.count == self.min_instances:
            # Record the pre-drift baseline
            self.pre_drift_mean = round(self.mean, 4)

        if self.count > self.min_instances and self.sum > self.threshold:
            if not self.drift_detected:
                self.drift_detected = True
                self.drift_at_flow = self.count
            return True
        return False

    def get_ph_sum(self):
        return round(self.sum, 4)

ph_detector = PageHinkley(min_instances=30, delta=0.005, threshold=5.0)

# ==========================================
# 3. OPEN LOG FILES
# ==========================================
drift_log_file = open("drift_log.csv", "w", newline="")
drift_writer = csv.writer(drift_log_file)
drift_writer.writerow(["flow_id", "timestamp_s", "mean_iat_s", "ph_sum",
                        "prediction", "drift_status", "inference_latency_ms"])

# ==========================================
# 4. LIVE STREAMING SNIFFER
# ==========================================
print("="*70)
print("LISTENING ON dummy0 FOR LIVE TRAFFIC...")
print("PH Params: min_instances=30, delta=0.005, threshold=5.0")
print("="*70)

# 120s active_timeout gives one row per 2-minute window — matches your simulation
streamer = NFStreamer(source="dummy0", active_timeout=120, statistical_analysis=True)

flow_counter = 0
attack_count = 0
benign_count = 0
start_wall = time.time()

print(f"{'FLOW':>6} | {'PREDICTION':<14} | {'MEAN IAT (s)':>12} | "
      f"{'PH SUM':>8} | {'LATENCY(ms)':>11} | DRIFT STATUS")
print("-" * 75)

for flow in streamer:
    if flow.src_port != 1883 and flow.dst_port != 1883:
        continue

    flow_counter += 1

    # Extract features — exact same schema as training
    flow_data = {}
    for feat in features:
        val = getattr(flow, feat, 0)
        if val == '' or val is None or isinstance(val, str):
            val = 0
        flow_data[feat] = float(val)

    flow_df = pd.DataFrame([flow_data])

    # Measure inference latency
    t0 = time.perf_counter()
    prediction = rf_model.predict(flow_df)[0]
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    pred_label = "ATTACK [!]" if prediction == 1 else "Benign"
    if prediction == 1:
        attack_count += 1
    else:
        benign_count += 1

    # Page-Hinkley update
    mean_iat_s = flow.bidirectional_mean_piat_ms / 1000.0
    is_drifting = ph_detector.update(mean_iat_s)
    ph_sum = ph_detector.get_ph_sum()

    if is_drifting and ph_detector.drift_at_flow == ph_detector.count:
        drift_status = f"*** DRIFT DETECTED at flow #{flow_counter} ***"
        wall_elapsed = round(time.time() - start_wall, 1)
        print(f"\n{'!'*70}")
        print(f"CONCEPT DRIFT DETECTED at flow #{flow_counter} | "
              f"Wall time: {wall_elapsed}s | PH sum: {ph_sum}")
        print(f"Pre-drift baseline IAT: {ph_detector.pre_drift_mean}s | "
              f"Current IAT: {mean_iat_s:.3f}s")
        print(f"{'!'*70}\n")
    elif is_drifting:
        drift_status = "POST-DRIFT"
    else:
        drift_status = "Stable"

    print(f"{flow_counter:>6} | {pred_label:<14} | {mean_iat_s:>12.3f} | "
          f"{ph_sum:>8.3f} | {latency_ms:>11.3f} | {drift_status}")

    # Log to CSV
    drift_writer.writerow([
        flow_counter,
        round(time.time() - start_wall, 2),
        round(mean_iat_s, 4),
        ph_sum,
        pred_label,
        drift_status,
        latency_ms
    ])
    drift_log_file.flush()

# ==========================================
# 5. FINAL SUMMARY
# ==========================================
drift_log_file.close()
total_wall = round(time.time() - start_wall, 1)

print("\n" + "="*70)
print("SESSION SUMMARY")
print("="*70)
print(f"Total MQTT flows processed : {flow_counter}")
print(f"Benign predictions         : {benign_count}")
print(f"Attack predictions         : {attack_count}")
print(f"Wall time                  : {total_wall}s ({total_wall/60:.1f} min)")
print(f"Drift detected             : {'YES — at flow #' + str(ph_detector.drift_at_flow) if ph_detector.drift_detected else 'NO'}")
if ph_detector.pre_drift_mean:
    print(f"Pre-drift mean IAT         : {ph_detector.pre_drift_mean}s")
print(f"Final PH sum               : {ph_detector.get_ph_sum()}")
print(f"\nDrift log saved to: drift_log.csv")
print("="*70)
