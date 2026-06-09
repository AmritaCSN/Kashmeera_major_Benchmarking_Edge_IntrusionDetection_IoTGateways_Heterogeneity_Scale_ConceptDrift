# REA-HID Benchmarking Framework

## Benchmarking Edge Intrusion Detection in IoT Gateways Under Heterogeneity, Scale, and Concept Drift

**Author:** Kashmeera R | AM.SC.P2CSN24006  
**Guide:** Shri. Hari N. N., Assistant Professor  
**Co-Guide:** Dr. Kurunandan Jain, Assistant Professor  
**Institution:** Center for Cybersecurity Systems and Networks, Amrita Vishwa Vidyapeetham, Amritapuri

**Building on:**
- [Replay-Enhanced Adaptive Hybrid Intrusion Detection (REA-HID) for IoT Edge Gateways](https://doi.org/10.1109/ICSEDIS68157.2026.11517953) — IEEE ICSEDIS 2026

---

## What This Project Does

This project introduces a **reproducible, live-streaming benchmarking framework** for evaluating Edge Machine Learning (EdgeML) Intrusion Detection Systems (IDS) deployed on resource-constrained IoT gateways.

Current public IoT IDS benchmarks consistently report F1-scores above 99%. **We prove these scores are illusory.** Models trained on standard corpora such as MQTTEEB-D and MQTT-IoT-IDS2020 exploit volumetric recording artifacts — raw packet sizes, absolute timestamps, and TCP time deltas — rather than learning genuine attack behaviour. We call this the **Accuracy Illusion**.

This framework addresses three failure modes that no existing benchmark tests simultaneously:

**1. Shortcut Learning (Predictive Invalidity)**  
We introduce the **Shortcut Dominance Score (SDS)** — the first scalar metric for quantifying shortcut learning risk in IoT IDS datasets. Applied across five public benchmarks, SDS reveals that three of four MQTT corpora are shortcut-dominated (SDS >= 0.60). MQTT-IoT-IDS2020 achieves F1=1.000 across three model architectures — a mathematical impossibility under genuine behavioral detection.

**2. Timing-Aware Evasion (The Detection Boundary)**  
Low-and-slow attacks that pace within the benign Inter-Arrival Time (IAT) envelope achieve 61.8% evasion against static ML classifiers, even after all volumetric shortcuts are eliminated. We prove that evasion succeeds if and only if the attack IAT distribution overlaps the benign distribution — a structural detection boundary not previously characterised in the literature.

**3. Hardware Deployment Ceiling**  
On a Raspberry Pi 4 running the full live IDS pipeline, per-flow inference latency reaches 207ms under 200 concurrent devices — 53x higher than the server baseline of 3.9ms. NFStream stateful RAM grows non-linearly from +15MB at 10 devices to +237MB at 200 devices. We establish a concrete deployment ceiling of approximately 578 concurrent devices before real-time classification guarantees are lost.

---

## Why This Project Is Useful

- Provides the **first quantitative scalar metric (SDS)** for dataset quality evaluation in IoT IDS research
- Produces the **first dual-protocol (MQTT + CoAP) benchmarking corpus** with enforced volumetric feature parity across all traffic classes
- Characterises the **detection boundary as a function of IAT distribution overlap** — a precise, falsifiable claim backed by four attack variants
- Establishes the **first empirical hardware deployment ceiling** for a live NFStream-based IDS on Raspberry Pi hardware
- Demonstrates that under volumetric feature parity, **model architecture is a resource variable, not a performance variable** (spread collapses from 0.147 to 0.011)

---

## Key Results

| Metric | Value | Significance |
|--------|-------|--------------|
| RF F1-Score (10-seed mean) | 0.9043 ± 0.0028 | Stable, architecture-independent baseline |
| RF Attack Recall | 0.7043 ± 0.0143 | Detection rate on volumetrically camouflaged attacks |
| Jitter Evasion Rate | 61.8% | IAT overlap with benign distribution |
| SlowPub Evasion Rate | 52.9% | Partial IAT overlap |
| Flood Evasion Rate | 0.0% | Distinctive burst signature detected |
| Sparse Evasion Rate | 0.0% | Protocol-only flow signature detected |
| Ablation: IAT-only (12 features) | F1 = 0.8982 | Drop of 0.006 vs 66 features — shortcuts carry no signal |
| MQTT-IDS2020 SDS | 1.00 | Entirely shortcut-dominated |
| REA-HID Dataset SDS | 0.00 | Behaviour-driven |
| Snort v2.9 on Jitter + SlowPub | 0 alerts | Signature-based detection structurally blind |
| Snort v2.9 on Nmap scan | 80,004 alerts | Baseline comparison |
| RPi 4 peak latency (200 devices) | 207ms | 53x above server baseline (3.9ms) |
| Hardware deployment ceiling | ~578 devices | Beyond this, real-time detection fails |

---

## Directory Structure

```
Kashmeera_major_Benchmarking_Edge_IntrusionDetection/
|
|-- README.md
|-- architecture_diagram.png
|-- docker-compose.yml
|-- docker-compose-scale.yml
|-- mosquitto.conf
|-- Dockerfile
|-- Dockerfile.coap
|
|-- data_generation/
|   |-- iot_device.py              # Benign MQTT IoT sensor simulator
|   |-- coap_device.py             # Benign CoAP device simulator
|   |-- extract_final_v1.1.py      # NFStream flow extractor (active/idle timeout=120s)
|   |-- label_final_v1.1.py        # IP-subnet ground-truth labeller
|
|-- audit/
|   |-- shortcut_audit_multi_v2.py # SDS metric — main contribution (Gini + SHAP + Permutation)
|   |-- audit_public_dataset.py    # Single-dataset bias audit utility
|
|-- evaluation/
|   |-- final_evaluation.py        # Multi-seed RF, feature ablation, McNemar test
|   |-- multi_model_benchmark.py   # RF vs XGBoost vs MLP comparison
|   |-- prove_evasion.py           # Per-attack evasion rates + SHAP analysis
|   |-- heterogeneity_proof.py     # MQTT + CoAP protocol-agnostic robustness proof
|   |-- train_baseline.py          # Trains and exports the baseline RF model
|
|-- streaming_and_hardware/
|   |-- streaming_listener_v2.py   # Live NFStream inference + Page-Hinkley drift detection
|   |-- monitor_resources.py       # RPi CPU / RAM / temperature profiler (psutil)
|   |-- verify_docker_iat.py       # Validates simulated IAT values from Docker
|
|-- pre_run_outputs/
|   |-- shortcut_audit_summary.csv # SDS scores across all 5 datasets
|   |-- evasion_report.csv         # Per-attack evasion percentages
|   |-- rpi_hardware_profile.csv   # Phase 4 RPi hardware metrics (200 devices)
|   |-- drift_log.csv              # Page-Hinkley drift detection events (live run)
|
|-- outputs/
    |-- shortcut_audit/            # Per-dataset Gini, SHAP, permutation figures + CSVs
    |-- final_eval/                # multiseed_results.csv, ablation_results.csv
    |-- model_benchmarks/          # multi_model_results.csv + comparison figure
    |-- figures/                   # evasion_reality.png, shap_bar.png, shap_summary.png
    |-- tables/                    # evasion_report.csv, shap_feature_importance.csv
```

---

## System Requirements

**Operating System:** Ubuntu 20.04 or later (tested on Ubuntu 22.04)

**Required Software:**
- Docker Engine 20.10+
- Docker Compose 1.29+
- Python 3.8+

**Python Libraries:**
```bash
pip install nfstream scikit-learn xgboost shap pandas numpy \
            matplotlib seaborn paho-mqtt statsmodels joblib river
```

**For Public Dataset Audit (optional):**  
Download the following datasets and place them at the paths shown:

| Dataset | Path |
|---------|------|
| MQTTEEB-D | `../data/raw/MQTTEEB-D_Final_Dataset/Preprocessed_Data/MQTTEEB-D_cleaned_data.csv` |
| MQTT-IoT-IDS2020 | `../data/raw/mqtt-iot-ids2020/` (folder containing biflow CSVs) |
| TON-IoT | `../data/raw/TON_IoT/ton_iot_network.csv` |
| CIC-IoT-2023 | `../data/raw/CIC_IoT_2023/cic_iot_2023.csv` |

If public datasets are not available, the shortcut audit will still run on the REA-HID dataset alone.

---

## Quick Start — Full Pipeline End to End

> **Note on pre-run outputs:** Due to physical hardware dependencies (Raspberry Pi 4) and the time required for a full 4-hour Docker simulation, key outputs are pre-generated in `pre_run_outputs/`. You can view results immediately without running the full simulation. Follow steps 5 onwards to run evaluation directly on the pre-labelled dataset.

### Step 1 — Clone the repository
```bash
git clone https://github.com/AmritaCSN/<repo-name>.git
cd <repo-name>
```

### Step 2 — Start the Docker simulation
```bash
docker-compose -f docker-compose-scale.yml up -d
```
This starts 40 MQTT benign devices, 6 CoAP devices, 12 attackers (3 per attack type), and the Mosquitto broker. Let it run for at least 30 minutes (4 hours for the full dataset).

### Step 3 — Extract flows from captured PCAP
```bash
python data_generation/extract_final_v1.1.py
# Output: final_mqtt_coap_flows.csv
```

### Step 4 — Label the flows
```bash
python data_generation/label_final_v1.1.py
# Output: dataset_mqtt_coap_final.csv (11,413 flows, 5 classes)
```

### Step 5 — Run the Shortcut Audit (main contribution)
```bash
python audit/shortcut_audit_multi_v2.py
# Output: outputs/shortcut_audit/shortcut_audit_combined.png
#         outputs/shortcut_audit/audit_summary.csv (SDS scores)
```

### Step 6 — Run final evaluation
```bash
python evaluation/final_evaluation.py
# Output: outputs/final_eval/multiseed_results.csv
#         outputs/final_eval/ablation_results.csv
#         outputs/final_eval/final_results_summary.png
```

### Step 7 — Run evasion proof and SHAP analysis
```bash
python evaluation/prove_evasion.py
# Output: outputs/tables/evasion_report.csv
#         outputs/figures/shap_bar.png
#         outputs/figures/evasion_reality.png
```

### Step 8 — Stop Docker
```bash
docker-compose -f docker-compose-scale.yml down
```

---

## Advanced Usage

### Adjusting Simulation Scale

Open `docker-compose-scale.yml` and modify the number of container replicas per service. Default configuration:

| Service | Subnet | Count | IAT Range |
|---------|--------|-------|-----------|
| Benign MQTT devices | 192.168.1.x | 40 | 30–60s (pre-drift), 50–90s (post-drift) |
| Benign CoAP devices | 192.168.1.x | 6 | 30–60s |
| Attack_Jitter | 10.0.0.x | 3 | 45 ± 15s (non-adaptive) |
| Attack_SlowPub | 10.0.0.x | 3 | 60–120s |
| Attack_Flood | 10.0.0.x | 3 | Burst (4 pkts) + 150–180s silence |
| Attack_Sparse | 10.0.0.x | 3 | 120–300s |

### Adjusting Flow Window Size

Open `data_generation/extract_final_v1.1.py` and modify:
```python
streamer = NFStreamer(
    source=pcap_file,
    statistical_analysis=True,
    active_timeout=120,   # change this — seconds per flow window
    idle_timeout=120      # change this — idle cutoff
)
```
Default is 120 seconds. A shorter window produces more flows but less IAT statistical information per flow.

### Adjusting SDS Metric Parameters

Open `audit/shortcut_audit_multi_v2.py` and modify:
```python
# k = number of top features to evaluate for shortcuts (default 5)
sds = shortcut_dominance_score(list(top15_gini.index), k=5)

# Dataset paths (lines 30–34)
MQTTEEB_D_PATH   = "../data/raw/MQTTEEB-D_Final_Dataset/..."
MQTT_IDS2020_DIR = "../data/raw/mqtt-iot-ids2020"
TON_IOT_PATH     = "../data/raw/TON_IoT/ton_iot_network.csv"
CIC_IOT_PATH     = "../data/raw/CIC_IoT_2023/cic_iot_2023.csv"
```

### Running the Live Streaming IDS on Raspberry Pi

On the RPi, with Mosquitto broker running:
```bash
# Terminal 1 — start hardware profiler
python streaming_and_hardware/monitor_resources.py
# Output: rpi_hardware_profile.csv (CPU, RAM, temperature every 2 seconds)

# Terminal 2 — start live IDS
sudo python streaming_and_hardware/streaming_listener_v2.py
# Output: ids_results.csv (per-flow prediction, IAT, Page-Hinkley value)
```

On the server, send traffic to the RPi broker:
```bash
# Benign traffic phase
python rpi_traffic_sender.py --broker <rpi-ip> --role benign --count 10 --duration 600

# Attack phase (run simultaneously in separate terminals)
python rpi_traffic_sender.py --broker <rpi-ip> --role jitter --count 3 --duration 600
python rpi_traffic_sender.py --broker <rpi-ip> --role flood  --count 3 --duration 600
```

### Key Parameters Reference

| Parameter | File | Default | Effect |
|-----------|------|---------|--------|
| `active_timeout` | extract_final_v1.1.py | 120s | Flow window size |
| `idle_timeout` | extract_final_v1.1.py | 120s | Flow idle cutoff |
| `n_estimators` | final_evaluation.py | 100 | RF tree count |
| `SEEDS` | final_evaluation.py | 10 seeds | Statistical stability evaluation |
| `SDS k` | shortcut_audit_multi_v2.py | 5 | Top-k features for SDS calculation |
| PH `delta` | streaming_listener_v2.py | 0.005 | Page-Hinkley sensitivity |
| PH `threshold` | streaming_listener_v2.py | 5.0 | Page-Hinkley alarm threshold |
| PH `min_instances` | streaming_listener_v2.py | 30 | Minimum flows before drift detection |

---

## System Architecture

See `architecture_diagram.png` for the full visual diagram. The framework has three layers:

**Layer 1 — Docker Simulation (Data Generation)**
```
Benign subnet (192.168.1.x)                      Attack subnet (10.0.0.x)
  40x MQTT sensors (iot_device.py)                  3x Jitter attackers
  6x  CoAP devices (coap_device.py)                 3x SlowPub attackers
          |                                         3x Flood attackers
          |                                         3x Sparse attackers
          |                                                   |
          |                                                   | 
          +---------> Mosquitto Broker (port 1883) <----------+
                               |
                          tcpdump capture
                               |
                          PCAP files
```

**Layer 2 — Flow Extraction and Labelling**
```
PCAP files
    |
    +--> extract_final_v1.1.py (NFStream, timeout=120s)
    |         --> final_mqtt_coap_flows.csv (unlabelled)
    |
    +--> label_final_v1.1.py (IP subnet ground truth)
              --> dataset_mqtt_coap_final.csv (11,413 labelled flows)
```

**Layer 3 — Evaluation Pipeline**
```
dataset_mqtt_coap_final.csv
    |
    +-- shortcut_audit_multi_v2.py  --> SDS scores, SHAP, Gini, Permutation
    |
    +-- final_evaluation.py         --> RF 0.9043±0.003, ablation 0.006 drop
    |
    +-- prove_evasion.py            --> Jitter 61.8%, SlowPub 52.9% evasion
    |
    +-- multi_model_benchmark.py    --> Architecture convergence (spread 0.011)
    |
    +-- heterogeneity_proof.py      --> Protocol-agnostic IAT dominance

Layer 4 — HITL Deployment (Raspberry Pi 4)
    |
    +-- streaming_listener_v2.py    --> Live inference, Page-Hinkley drift
    |
    +-- monitor_resources.py        --> CPU/RAM/temp profiling
              --> 207ms peak latency at 200 devices
              --> ~578-device deployment ceiling
```

---

## Shortcut Dominance Score (SDS) — The Core Metric

SDS is defined as:

```
SDS = (number of top-k features that are volumetric or temporal leakage artefacts) / k
```

Where:
- **Volumetric artefacts:** raw packet sizes, byte counts, packet counts
- **Temporal leakage artefacts:** absolute timestamps, flow duration, TCP time delta
- **k = 5** (default) — top 5 features by Gini importance

| SDS Value | Interpretation |
|-----------|----------------|
| 0.00 | Behaviour-driven — model learns genuine temporal signals |
| 0.40–0.59 | Mixed — partial shortcut reliance |
| 0.60–1.00 | Shortcut-dominated — high-accuracy scores are illusory |

---

## Attack Taxonomy

| Attack | IAT Range | Evasion Rate | Detection Feature | Key Property |
|--------|-----------|--------------|-------------------|--------------|
| Attack_Jitter | 30–60s | 61.8% | bidirectional_mean_piat_ms | Overlaps benign IAT by design — does not adapt post-drift |
| Attack_SlowPub | 60–120s | 52.9% | dst2src_mean_piat_ms | Overlaps post-drift benign baseline (50–90s) |
| Attack_Flood | Burst + 150–180s | 0.0% | bidirectional_stddev_piat_ms | Burst creates distinctive standard deviation signature |
| Attack_Sparse | 120–300s | 0.0% | bidirectional_packets | Protocol-only flow — no PUBLISH packets in 120s window |

**Core structural finding:** Evasion succeeds if and only if the attack IAT distribution overlaps with the benign distribution at detection time. This is the fundamental detection boundary of snapshot-based classification.

---

## Hardware Deployment Results (Raspberry Pi 4)

| Phase | Concurrent Devices | RAM Increase | CPU Peak | Max Latency | Temperature |
|-------|--------------------|--------------|----------|-------------|-------------|
| 1 — Benign baseline | 10 | +15 MB | 31.4% | 139.9 ms | 48.9°C |
| 2 — Mixed (+ attackers) | 16 | +8 MB | 31.7% | 93.9 ms | 49.4°C |
| 3 — Scale | 50+ | +21 MB | 30.8% | 112.4 ms | 49.2°C |
| 4 — Stress test | 200 | +237 MB | 31.4% | 207.3 ms | 52.1°C |

**Server baseline:** 3.9ms per-flow inference  
**RPi at 200 devices:** 207ms = **53x above server baseline**  
**Deployment ceiling:** ~578 concurrent devices before real-time classification fails  
**Key insight:** CPU is never the bottleneck. RAM growth and latency scaling are the real constraints.

---

## Published Work

1. **K. R, P. Madhu, V. K K, D. Rajeev, K. Jain, and P. Krishnan**, "Time-based Trojan Detection using Ensemble Learning," in *Proc. IEEE ICIMIA 2025*, Sep. 2025.  
   DOI: [10.1109/ICIMIA67127.2025.11200592](https://doi.org/10.1109/ICIMIA67127.2025.11200592)

2. **K. R, P. Krishnan, H. N. N, S. Subramanian N, and K. Jain**, "Replay-Enhanced Adaptive Hybrid Intrusion Detection (REA-HID) for IoT Edge Gateways," in *Proc. IEEE ICSEDIS 2026*, DOI: [10.1109/ICSEDIS68157.2026.11517953](https://doi.org/10.1109/ICSEDIS68157.2026.11517953)


---

## Limitations and Honest Notes

- The Docker simulation uses controlled synthetic traffic. Real-world IoT deployments will have additional protocol diversity.
- Hardware profiling measured full system RAM via psutil, not NFStream process-specific RAM. The +237MB figure is a conservative upper bound.
- The concept drift result (F1 drop of 0.074) is based on an imbalanced temporal split (1,407 pre-drift vs 10,006 post-drift flows). This is preliminary evidence, not a validated result.
- The Page-Hinkley detector uses fixed parameters (delta=0.005, threshold=5.0). Threshold calibration is an open research problem.
