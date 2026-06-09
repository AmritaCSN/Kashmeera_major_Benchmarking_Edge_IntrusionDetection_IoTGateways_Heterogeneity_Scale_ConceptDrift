import pandas as pd

# Load the trial flows
df = pd.read_csv("REA_HID_Publication_Dataset.csv")

# Filter out empty TCP ACKs and the Linux background noise (192.168.122.x)
df = df[df['bidirectional_packets'] > 2].copy()
df = df[df['src_ip'].str.startswith(('192.168.1.', '10.0.0.'))]
# note : the full  dataset ocntains some flows with src_ip
# Map the exact IPs to their Roles based on our docker-compose.yml
def get_role(ip):
    if ip.startswith('192.168.1.'): return 'Benign (Target: 30-60s)'
    elif ip == '10.0.0.11': return 'SlowITe (Target: 60-120s)'
    elif ip == '10.0.0.12': return 'Jitter (Target: 30-60s)'
    elif ip == '10.0.0.13': return 'Flood Variant (Target: 5-10s)'
    elif ip == '10.0.0.14': return 'Sparse (Target: 120-300s)'
    return 'Unknown'

df['role'] = df['src_ip'].apply(get_role)

print("=== Mean IAT Verification by Role ===")
# Convert ms to seconds for easier reading

stats = df.groupby('role')['bidirectional_mean_piat_ms'].mean() / 1000 
print(stats.sort_values())

print("\n=== Packet Count Verification ===")
pkt_stats = df.groupby('role')['bidirectional_packets'].mean()
print(pkt_stats.sort_values())