"""
label_final.py
==============
Labels NFStream flow extracts using IP-subnet ground truth.

Ground truth methodology: In this controlled synthetic benchmark,
attacker IPs (10.0.0.x) are known by design. This is analogous to
NSL-KDD and BoT-IoT dataset labeling — a deliberate controlled evaluation
environment, not a claim of real-world deployment.

Attack taxonomy:
  Attack_SlowPub : Low-rate pacing attack (60-120s IAT).
                   NOTE: This is a sparse-publishing pacing attack.
                   It is inspired by but distinct from Vaccari et al.
                   SlowITe (Sensors 2020), which exploits MQTT KeepAlive
                   for broker connection slot exhaustion.
  Attack_Jitter  : Jitter-based evasion (45±15s, does not drift)
  Attack_Flood   : Micro-flood burst (4 packets then silence)
  Attack_Sparse  : Ultra-sparse publisher (120-300s IAT)
"""
import pandas as pd

print("Loading unlabeled flows...")
df = pd.read_csv("final_mqtt_coap_flows.csv")

# Sparse attacks may have very few packets per window — keep flows >= 2 packets
df = df[df['bidirectional_packets'] >= 2].copy()

def assign_label(row):
    src = str(row['src_ip'])
    dst = str(row['dst_ip'])

    attack_ips = {
        '10.0.0.11': 'Attack_SlowPub',
        '10.0.0.15': 'Attack_SlowPub',
        '10.0.0.16': 'Attack_SlowPub',
        '10.0.0.12': 'Attack_Jitter',
        '10.0.0.17': 'Attack_Jitter',
        '10.0.0.18': 'Attack_Jitter',
        '10.0.0.13': 'Attack_Flood',
        '10.0.0.19': 'Attack_Flood',
        '10.0.0.20': 'Attack_Flood',
        '10.0.0.14': 'Attack_Sparse',
        '10.0.0.21': 'Attack_Sparse',
        '10.0.0.22': 'Attack_Sparse',
    }

    if src in attack_ips:
        return 1, attack_ips[src]
    if dst in attack_ips:
        return 1, attack_ips[dst]

    if src.startswith('192.168.1.') or dst.startswith('192.168.1.'):
        return 0, 'Benign'

    return -1, 'Noise'


df[['label', 'attack_type']] = df.apply(assign_label, axis=1, result_type='expand')
df = df[df['label'] != -1]

print("\n=== Final Dataset Distribution ===")
print(df['attack_type'].value_counts())
print(f"\nTotal flows: {len(df)}")
print(f"Benign: {(df['label']==0).sum()} | Attack: {(df['label']==1).sum()}")

df.to_csv("dataset_mqtt_coap_final.csv", index=False)
print("\n[OK] Saved: dataset_mqtt_coap_final.csv")
