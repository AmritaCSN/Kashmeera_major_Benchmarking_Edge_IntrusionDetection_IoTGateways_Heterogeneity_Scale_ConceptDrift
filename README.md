Author: Kashmeera R | AM.SC.P2CSN24006

Guide: Shri. Hari N. N. | Co-Guide: Dr. Kurunandan Jain

Institution: Center for Cybersecurity Systems and Networks, Amrita Vishwa Vidyapeetham

🎯 What This Project Does

This project introduces a reproducible benchmarking framework for evaluating Edge Machine Learning (EdgeML) Intrusion Detection Systems (IDS) under three simultaneous stress conditions that current benchmarks ignore:

Shortcut Learning: Public IoT IDS datasets inflate model accuracy by containing volumetric artifacts (packet sizes, timestamps). We introduce the Shortcut Dominance Score (SDS), the first scalar metric for quantifying this problem.

Timing-Aware Evasion: Low-and-slow attacks that pace within the benign Inter-Arrival Time (IAT) envelope achieve high evasion against static ML classifiers when volumetric shortcuts are eliminated.

Hardware Deployment Ceiling: Real-time evaluation of the IDS pipeline on a Raspberry Pi 4 to empirically establish deployment latency and hardware exhaustion limits.

💡 Why This Project Is Useful

Public IoT IDS benchmarks consistently report >99% F1-scores. We prove these scores are mathematically illusory. Our framework:

📉 Provides the first quantitative metric (SDS) for dataset quality in IoT IDS.

🔄 Produces an MQTT+CoAP dual-protocol benchmark with enforced volumetric feature parity.

💻 Establishes the empirical hardware deployment ceiling for live NFStream-based IDS on constrained Edge computing hardware.

📊 Key Highlights & Results

Evaluation Metric

Measured Value

Significance

RF F1-Score (10-seed)

0.9043 ± 0.0028

Baseline performance based strictly on temporal flow physics.

Jitter Evasion Rate

61.8%

Proves static models fail catastrophically against low-and-slow pacing.

REA-HID SDS Score

0.00

Demonstrates behavior-driven detection (vs. Public Dataset SDS = 1.00).

RPi Peak Latency

207ms

Live edge inference ceiling at 200 concurrent connected devices.

📂 Directory Structure & File Descriptions

📦 Kashmeera_major_Benchmarking_Edge_IntrusionDetection_IoTGateways_Heterogeneity_Scale_ConceptDrift
├── 📁 audit/                      # Dataset bias analysis & shortcut quantification
│   ├── 🐍 audit_public_dataset.py # Evaluates a single public dataset for volumetric bias
│   └── 🐍 shortcut_audit_multi_v2.py # Calculates SDS across multiple datasets (SHAP/Gini)
│
├── 📁 data_generation/            # Docker IoT simulation & flow extraction
│   ├── 🐍 coap_device.py          # Simulates benign CoAP protocol traffic
│   ├── 🐍 extract_final_v1.1.py   # Extracts stateful network flows via NFStream
│   ├── 🐍 iot_device.py           # Simulates benign MQTT IoT sensor telemetry
│   └── 🐍 label_final_v1.1.py     # Applies ground-truth IP subnet labels
│
├── 📁 evaluation/                 # Model training & evasion testing
│   ├── 🐍 final_evaluation.py     # Multi-seed Random Forest training & ablation tests
│   ├── 🐍 heterogeneity_proof.py  # Validates robustness across MQTT/CoAP mixes
│   ├── 🐍 multi_model_benchmark.py# Compares baseline F1 (RF vs XGBoost vs MLP)
│   ├── 🐍 prove_evasion.py        # Tests static models against low-and-slow attacks
│   └── 🐍 train_baseline.py       # Trains and exports the baseline RF model
│
├── 📁 pre_run_outputs/            # Pre-generated logs for live Code Review / HITL
│   ├── 📄 drift_log.csv           # Live Page-Hinkley drift detection triggers
│   ├── 📄 evasion_report.csv      # Calculated evasion percentages (Jitter, SlowPub, Flood)
│   ├── 📄 rpi_hardware_profile.csv# CPU, RAM, and latency metrics from Raspberry Pi
│   └── 📄 shortcut_audit_summary.csv # Final calculated SDS scores (REA-HID vs Public)
│
├── 📁 streaming_and_hardware/     # Live Edge Gateway Deployment (Raspberry Pi)
│   ├── 🐍 monitor_resources.py    # Real-time CPU/Memory exhaustion profiler
│   ├── 🐍 streaming_listener_v2.py# Live flow extraction & Page-Hinkley inference
│   └── 🐍 verify_docker_iat.py    # Validates simulated IAT directly from Docker
│
└── ⚙️ Configuration Files         # docker-compose.yml, Dockerfiles, mosquitto.conf


💻 System Requirements & Installation

OS: Ubuntu 20.04+ (Tested on Ubuntu 22.04)

Dependencies: Docker + Docker Compose

Environment: Python 3.8+

Python Libraries:

pip install nfstream scikit-learn xgboost shap pandas numpy matplotlib seaborn paho-mqtt river


🚀 Quick Start — Run Everything End to End

⚠️ Note on Hardware-in-the-Loop (HITL) Execution: Due to the physical hardware dependencies (Raspberry Pi 4) and time constraints of a live code review, the live streaming pipeline (streaming_listener_v2.py) and hardware profiler (monitor_resources.py) outputs have been safely pre-generated and are available in the pre_run_outputs/ directory.

1. Clone the repository

git clone [https://github.com/Kashmeerars/Kashmeera_major_Benchmarking_Edge_IntrusionDetection_IoTGateways_Heterogeneity_Scale_ConceptDrift.git](https://github.com/Kashmeerars/Kashmeera_major_Benchmarking_Edge_IntrusionDetection_IoTGateways_Heterogeneity_Scale_ConceptDrift.git)
cd Kashmeera_major_Benchmarking_Edge_IntrusionDetection_IoTGateways_Heterogeneity_Scale_ConceptDrift


2. Start the Docker Simulation (Generates realistic device traffic)

docker-compose -f docker-compose-scale.yml up -d


3. Evaluate the Public Datasets (The Shortcut Audit)

python audit/shortcut_audit_multi_v2.py


4. Run Adversarial Evasion Tests

python evaluation/prove_evasion.py


⚙️ Advanced Usage & Parameter Tuning

You can modify the behavior of the framework by adjusting key parameters:

Adjusting the Simulation Scale: Open docker-compose-scale.yml to change device counts. Default is 40 MQTT benign + 6 CoAP + 12 attackers.

Feature Window Size: Open data_generation/extract_final_v1.1.py. Adjust active_timeout=120s to change the NFStream flow cut-off limits.

Hardware Profiling: To run the physical deployment on an RPi, execute python streaming_and_hardware/monitor_resources.py alongside the streaming listener.

🏗️ System Architecture

Please refer to architecture_diagram.png in the repository root for a high-level overview detailing the three core modules: The Docker Simulation (192.168.1.x/10.0.0.x subnets), the Flow Extraction Pipeline (NFStream), and the Evaluation/HITL Inference Gateway.REA-HID Benchmarking Framework

Benchmarking Edge Intrusion Detection in IoT Gateways Under Heterogeneity, Scale, and Concept Drift

Author: Kashmeera R | AM.SC.P2CSN24006

Guide: Shri. Hari N. N. | Co-Guide: Dr. Kurunandan Jain

Institution: Center for Cybersecurity Systems and Networks, Amrita Vishwa Vidyapeetham

What This Project Does

This project introduces a reproducible benchmarking framework for evaluating Edge Machine Learning (EdgeML) Intrusion Detection Systems (IDS) under three simultaneous stress conditions that current benchmarks ignore:

Shortcut Learning: Public IoT IDS datasets inflate model accuracy by containing volumetric artifacts (packet sizes, timestamps). We introduce the Shortcut Dominance Score (SDS), the first scalar metric for quantifying this problem.

Timing-Aware Evasion: Low-and-slow attacks that pace within the benign Inter-Arrival Time (IAT) envelope achieve high evasion against static ML classifiers when volumetric shortcuts are eliminated.

Hardware Deployment Ceiling: Real-time evaluation of the IDS pipeline on a Raspberry Pi 4 to empirically establish deployment latency and hardware exhaustion limits.

Why This Project Is Useful

Public IoT IDS benchmarks report >99% F1-scores. We prove these scores are mathematically illusory. Our framework:

Provides the quantitative metric (SDS) for dataset quality in IoT IDS.

Produces an MQTT+CoAP dual-protocol benchmark with enforced volumetric feature parity.

Establishes the empirical hardware deployment ceiling for live NFStream-based IDS on constrained Edge computing hardware.

Directory Structure and File Descriptions

audit/

Contains scripts to analyze dataset bias and quantify shortcut learning in public IoT IDS datasets.

audit_public_dataset.py: Evaluates a single public dataset to expose its volumetric bias and feature importance.

shortcut_audit_multi_v2.py: Calculates the Shortcut Dominance Score (SDS) across multiple datasets using SHAP and Gini importance.

data_generation/

Manages the Docker-based IoT simulation and NFStream flow extraction pipeline.

coap_device.py: Simulates benign CoAP protocol traffic for protocol heterogeneity validation.

extract_final_v1.1.py: Uses NFStream to extract stateful network flows from captured PCAP files.

iot_device.py: Simulates benign MQTT IoT sensor telemetry (temperature/humidity).

label_final_v1.1.py: Applies ground-truth labels to extracted flows based on IP subnet configurations.

evaluation/

Contains scripts for model training, baseline benchmarking, and testing evasion resilience.

final_evaluation.py: Runs multi-seed Random Forest training and feature ablation tests.

heterogeneity_proof.py: Validates model performance and robustness across mixed MQTT and CoAP protocol traffic.

multi_model_benchmark.py: Compares baseline F1 scores across Random Forest, XGBoost, and MLP architectures.

prove_evasion.py: Tests static models against low-and-slow volumetric camouflage attacks to calculate exact evasion rates.

train_baseline.py: Trains and exports the standard baseline Random Forest model.

pre_run_outputs/

Stores pre-generated logs and results for the Hardware-in-the-Loop (HITL) and audit pipelines (used for the live code review).

drift_log.csv: Logs the real-time Page-Hinkley drift detection triggers from the streaming listener.

evasion_report.csv: Contains the calculated evasion percentages for Jitter, SlowPub, and Flood attacks.

rpi_hardware_profile.csv: Logs CPU, RAM, and latency metrics from the Raspberry Pi edge deployment.

shortcut_audit_summary.csv: Stores the final calculated SDS scores comparing REA-HID against public datasets.

streaming_and_hardware/

Contains the scripts meant for live deployment on the Edge Gateway (Raspberry Pi).

monitor_resources.py: Profiles CPU and memory exhaustion metrics in real-time during live inference.

streaming_listener_v2.py: Runs live flow extraction and stateful inference using Page-Hinkley drift detection.

verify_docker_iat.py: Validates the simulated Inter-Arrival Times (IAT) directly from the Docker interfaces.

Root Configuration Files

docker-compose.yml / docker-compose-scale.yml: Defines the benign and adversarial IoT simulation topology.

Dockerfile / Dockerfile.coap: Container configurations for the MQTT and CoAP simulated devices.

mosquitto.conf: Configuration file for the Eclipse Mosquitto broker.

System Requirements & Installation

OS: Ubuntu 20.04+

Dependencies: Docker + Docker Compose

Environment: Python 3.8+

Python Libraries:

pip install nfstream scikit-learn xgboost shap pandas numpy matplotlib seaborn paho-mqtt


Quick Start — Run Everything End to End

Note: Due to the physical hardware dependencies (Raspberry Pi 4) and time constraints of the live code review, the live streaming pipeline and hardware profiler outputs have been pre-generated and are available in the pre_run_outputs/ directory.

1. Clone the repository

git clone [https://github.com/Kashmeerars/Kashmeera_major_Benchmarking_Edge_IntrusionDetection_IoTGateways_Heterogeneity_Scale_ConceptDrift.git](https://github.com/Kashmeerars/Kashmeera_major_Benchmarking_Edge_IntrusionDetection_IoTGateways_Heterogeneity_Scale_ConceptDrift.git)
cd Kashmeera_major_Benchmarking_Edge_IntrusionDetection_IoTGateways_Heterogeneity_Scale_ConceptDrift


2. Start the Docker Simulation (Generates realistic device traffic)

docker-compose -f docker-compose-scale.yml up -d


3. Evaluate the Public Datasets (The Shortcut Audit)

python audit/shortcut_audit_multi_v2.py


4. Run Adversarial Evasion Tests

python evaluation/prove_evasion.py


Advanced Usage & Parameter Tuning

You can modify the behavior of the framework by adjusting key parameters:

Adjusting the Simulation Scale: Open docker-compose-scale.yml to change device counts. Default is 40 MQTT benign + 6 CoAP + 12 attackers.

Feature Window Size:
Open data_generation/extract_final_v1.1.py. Adjust active_timeout=120s to change the NFStream flow cut-off limits.

Hardware Profiling:
To run the physical deployment on an RPi, execute python streaming_and_hardware/monitor_resources.py alongside the streaming listener.

System Architecture

Please refer to architecture_diagram.png in the repository root for a high-level overview detailing the three core modules: The Docker Simulation (192.168.1.x/10.0.0.x subnets), the Flow Extraction Pipeline (NFStream), and the Evaluation/HITL Inference Gateway.
