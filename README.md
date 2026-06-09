# REA-HID Benchmarking Framework
## Benchmarking Edge Intrusion Detection in IoT Gateways Under Heterogeneity, Scale, and Concept Drift

**Author:** Kashmeera R | AM.SC.P2CSN24006  
**Guide:** Shri. Hari N. N. | **Co-Guide:** Dr. Kurunandan Jain  
**Institution:** Center for Cybersecurity Systems and Networks, Amrita Vishwa Vidyapeetham

---

## What This Project Does
This project introduces a reproducible benchmarking framework for evaluating Edge Machine Learning (EdgeML) Intrusion Detection Systems (IDS) under three simultaneous stress conditions that current benchmarks ignore:

1. **Shortcut Learning:** Public IoT IDS datasets inflate model accuracy by containing volumetric artifacts (packet sizes, timestamps). We introduce the **Shortcut Dominance Score (SDS)**, the first scalar metric for quantifying this problem.
2. **Timing-Aware Evasion:** Low-and-slow attacks that pace within the benign Inter-Arrival Time (IAT) envelope achieve high evasion against static ML classifiers when volumetric shortcuts are eliminated.
3. **Hardware Deployment Ceiling:** Real-time evaluation of the IDS pipeline on a Raspberry Pi 4 to empirically establish deployment latency and hardware exhaustion limits.

---

## Why This Project Is Useful
Public IoT IDS benchmarks report >99% F1-scores. We prove these scores are mathematically illusory. Our framework:
- Provides the quantitative metric (SDS) for dataset quality in IoT IDS.
- Produces an MQTT+CoAP dual-protocol benchmark with enforced volumetric feature parity.
- Establishes the empirical hardware deployment ceiling for live NFStream-based IDS on constrained Edge computing hardware.

---

## System Requirements & Installation
- **OS:** Ubuntu 20.04+ 
- **Dependencies:** Docker + Docker Compose
- **Environment:** Python 3.8+ 

**Python Libraries:**
```bash
pip install nfstream scikit-learn xgboost shap pandas numpy matplotlib seaborn paho-mqtt
