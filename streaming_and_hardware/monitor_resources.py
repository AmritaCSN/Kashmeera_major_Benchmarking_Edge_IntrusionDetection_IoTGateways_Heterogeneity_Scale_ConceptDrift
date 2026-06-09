"""
monitor_resources.py
====================
Hardware exhaustion profiling for the major project.
Profile: 4GB RAM (Raspberry Pi 4B simulation) & 2 vCPUs.

Usage:
    python monitor_resources.py --duration 1800 --output outputs/hardware_profiles/
"""

import psutil
import time
import csv
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=1800,
                    help="Monitoring duration in seconds (default 1800 = 30 mins)")
parser.add_argument("--interval", type=float, default=0.5,
                    help="Sample interval in seconds (default 0.5)")
parser.add_argument("--output", type=str, default="outputs/hardware_profiles/",
                    help="Output directory for CSV and PNG")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

csv_path = os.path.join(args.output, "resource_log_attack.csv")
png_path = os.path.join(args.output, "hardware_profile_4GB_attack.png")

print(f"[*] Hardware profiling started. Duration: {args.duration}s ({args.duration/60:.0f} min)")
print(f"[*] Sample interval: {args.interval}s | Output: {args.output}")
print(f"[*] RAM Profile: 4GB (Raspberry Pi 4B) | CPU limit: 2 vCPUs")
print("[*] Press Ctrl+C to stop early — data will still be saved.\n")

timestamps  = []
cpu_vals    = []
ram_mb_vals = []

start = time.time()

try:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s", "cpu_percent", "ram_mb"])

        while (time.time() - start) < args.duration:
            elapsed = round(time.time() - start, 2)
            cpu     = psutil.cpu_percent(interval=None)
            
            try:
                with open('/sys/fs/cgroup/memory.current', 'r') as mem_file:
                    ram_bytes = int(mem_file.read().strip())
                ram_mb = round(ram_bytes / 1e6, 1)
            except Exception:
                mem = psutil.virtual_memory()
                ram_mb = round(mem.used / 1e6, 1)

            writer.writerow([elapsed, cpu, ram_mb])
            f.flush()

            timestamps.append(elapsed)
            cpu_vals.append(cpu)
            ram_mb_vals.append(ram_mb)

            # Live console update every 30 seconds
            if len(timestamps) % 60 == 0:
                print(f"  t={elapsed:>7.0f}s | CPU: {cpu:>5.1f}% | RAM: {ram_mb:>7.1f}MB")

            time.sleep(args.interval)

except KeyboardInterrupt:
    print("\n[*] Interrupted. Saving results...")

print(f"\n[*] Collected {len(timestamps)} samples over {round(timestamps[-1]/60, 1) if timestamps else 0} minutes")

# ── PLOT ────────────────────────────────────────────────────────
if len(timestamps) < 5:
    print("[!] Not enough data to plot.")
    exit()

t_min = [t / 60.0 for t in timestamps]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("REA-HID Hardware Profiling under Distributed Pacing Attack\n"
             "(Simulated Raspberry Pi 4B Profile: 2 vCPU, 4GB RAM)",
             fontsize=13, fontweight="bold")

# CPU plot
ax1.plot(t_min, cpu_vals, color="#2E75B6", linewidth=1.5, alpha=0.8, label="CPU Usage (%)")
ax1.axhline(y=80, color="#C00000", linestyle="--", alpha=0.7, linewidth=1.5, label="80% Hardware Warning")
ax1.fill_between(t_min, cpu_vals, alpha=0.15, color="#2E75B6")
ax1.set_ylabel("CPU Usage (%)", fontsize=11)
ax1.set_ylim(0, 100)
ax1.legend(loc="upper right")
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# RAM plot
ax2.plot(t_min, ram_mb_vals, color="#C00000", linewidth=2.0, alpha=0.9, label="System RAM (MB)")
ax2.axhline(y=4096, color="black", linestyle="--", alpha=0.8, linewidth=2.0, label="4GB Pi 4B Target Limit (4096 MB)")
ax2.fill_between(t_min, ram_mb_vals, alpha=0.15, color="#C00000")
ax2.set_ylabel("RAM Usage (MB)", fontsize=11)
ax2.set_xlabel("Elapsed Time (minutes)", fontsize=11)
ax2.set_ylim(0, 5000) # Framed cleanly to show the stable 4.15GB baseline
ax2.legend(loc="lower right")
ax2.grid(axis="y", linestyle="--", alpha=0.4)

# Annotate peak values
if cpu_vals:
    peak_cpu = max(cpu_vals)
    peak_cpu_t = t_min[cpu_vals.index(peak_cpu)]
    ax1.annotate(f"Peak CPU: {peak_cpu:.1f}%",
                 xy=(peak_cpu_t, peak_cpu),
                 xytext=(peak_cpu_t + 0.5, peak_cpu + 10),
                 fontsize=10, color="#2E75B6", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#2E75B6"))

plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"[*] Graph saved: {png_path}")
print(f"[*] CSV saved:   {csv_path}")

# ── SUMMARY STATS ───────────────────────────────────────────────
print("\n── Resource Summary ──────────────────────────────────")
print(f"CPU  | Mean: {sum(cpu_vals)/len(cpu_vals):.1f}% | Peak: {max(cpu_vals):.1f}%")
print(f"RAM  | Mean: {sum(ram_mb_vals)/len(ram_mb_vals):.0f}MB | Peak: {max(ram_mb_vals):.0f}MB")
print("──────────────────────────────────────────────────────")