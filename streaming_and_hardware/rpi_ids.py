import joblib, time, csv
import numpy as np, pandas as pd
from nfstream import NFStreamer
from river.drift import PageHinkley

INTERFACE  = 'eth0'
MQTT_PORT  = 1883
MODEL_FILE = '/home/dell/rea-hid/rea_hid_rf_model.pkl'
FEAT_FILE  = '/home/dell/rea-hid/rea_hid_feature_names.pkl'
LOG_FILE   = '/home/dell/rea-hid/ids_results.csv'

model    = joblib.load(MODEL_FILE)
features = joblib.load(FEAT_FILE)
ph       = PageHinkley(min_instances=30, delta=0.005, threshold=5.0)

def get_ph_sum(ph):
    for attr in ('_sum', 'sum', 'cumsum', '_x_mean'):
        if hasattr(ph, attr):
            v = getattr(ph, attr)
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0

flow_count = attack_count = drift_count = 0
latencies  = []
start      = time.time()

print(f"[REA-HID] Model loaded | {len(features)} features | classes={list(model.classes_)}")
print(f"[REA-HID] Listening on {INTERFACE} port {MQTT_PORT}")
print(f"{'FLOW':>6} | {'PRED':>5} | {'LABEL':<12} | {'IAT(s)':>8} | {'PH':>6} | {'LAT(ms)':>8} | STATUS")
print("-" * 75)

with open(LOG_FILE, 'w', newline='') as logf:
    writer = csv.writer(logf)
    writer.writerow(['flow_id','wall_time_s','src_ip','mean_iat_s',
                     'ph_sum','prediction','latency_ms','drift'])

    streamer = NFStreamer(source=INTERFACE,
                         statistical_analysis=True,
                         active_timeout=120,
                         idle_timeout=120)

    for flow in streamer:
        if flow.dst_port != MQTT_PORT and flow.src_port != MQTT_PORT:
            continue
        if flow.bidirectional_packets < 2:
            continue

        flow_count += 1
        t0 = time.perf_counter()

        # Build named DataFrame — eliminates feature names warning
        row = {f: getattr(flow, f, np.nan) for f in features}
        X   = pd.DataFrame([row])[features].fillna(0)
        pred    = model.predict(X)[0]
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        iat = getattr(flow, 'bidirectional_mean_piat_ms', 0) / 1000.0
        ph.update(iat)
        drifted  = ph.drift_detected
        ph_value = get_ph_sum(ph)
        if drifted:
            drift_count += 1

        is_attack = (int(pred) == 1)
        if is_attack:
            attack_count += 1

        label = 'ATTACK [!]' if is_attack else 'benign'
        src   = getattr(flow, 'src_ip', '?')

        print(f"{flow_count:>6} | {int(pred):>5} | {label:<12} | {iat:>8.3f} | "
              f"{ph_value:>6.3f} | {latency:>8.3f} | "
              f"{'*** DRIFT ***' if drifted else 'Stable'}")

        writer.writerow([flow_count, round(time.time()-start, 2), src,
                         round(iat, 4), round(ph_value, 4),
                         int(pred), round(latency, 3),
                         'DRIFT' if drifted else 'Stable'])
        logf.flush()

        if flow_count % 10 == 0:
            mean_lat = np.mean(latencies[-10:])
            print(f"  >> flows={flow_count} | attacks={attack_count} | "
                  f"drifts={drift_count} | mean_lat={mean_lat:.2f}ms")