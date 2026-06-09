from nfstream import NFStreamer
import pandas as pd
import sys
import os

#active_timeout=600 → 120: Creates one flow record every 2 minutes instead of every 10. This alone will give you ~5x more rows from the same PCAP file.
#idle_timeout=600 → 120: Sparse attacks that go quiet for long periods will now get properly closed and recorded rather than being held open in one giant window.
#Added sys.argv support: Now you can call python extract_final.py simulation_scale.pcap for the scale run instead of editing the file each time.
# Accept PCAP filename as argument, default to scale simulation output
pcap_file = sys.argv[1] if len(sys.argv) > 1 else "final_mqtt_coap_sim.pcap"

if not os.path.exists(pcap_file):
    print(f"[ERROR] PCAP file not found: {pcap_file}")
    print("Usage: python extract_final.py <pcap_filename>")
    sys.exit(1)

print(f"Extracting flows from: {pcap_file}")
print("Window size: 120s (produces ~5x more flows than 600s window)")

streamer = NFStreamer(
    source=pcap_file,
    statistical_analysis=True,
    active_timeout=120,
    idle_timeout=120
)

df = streamer.to_pandas()

target_flows = df[
    (df['src_port'] == 1883) | (df['dst_port'] == 1883) |
    (df['src_port'] == 5683) | (df['dst_port'] == 5683)
].copy()

print(f"Total flows extracted: {len(df)}")
print(f"MQTT & CoAP flows only: {len(target_flows)}")
target_flows.to_csv("final_mqtt_coap_flows.csv", index=False)
print("Saved to final_mqtt_coap_flows.csv")
