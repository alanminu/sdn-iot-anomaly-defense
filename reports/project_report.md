# AI-Driven Anomaly Detection and Mitigation Framework for IoT Traffic in Software-Defined Networks

**Team:** Alan Minu (24BDS0197) · Harsh Choudhary (24BDS0272) · Gopalkrishna (24BCE2901) · Samarth Sahu (24BDS0126)

---

## 1. Literature Review

Ten papers published 2024–2026 were surveyed to position this project relative to the current state of the art in SDN-based IoT intrusion detection and mitigation.

1. **AI-Driven Intrusion Detection and Mitigation Framework for Software-Defined IoT Networks** — *Peer-to-Peer Networking and Applications*, 2025. Proposes a combined detection-and-mitigation pipeline for SD-IoT networks — the closest existing work to this project's own title. Because the scope overlaps heavily, this project differentiates itself with a lighter model (Random Forest / XGBoost, sub-millisecond inference) and a faster detect-to-block response time.
2. **RNN-RYU: SDN-Enabled IoT Transport-Layer DDoS Attack Detection Using Recurrent Neural Networks** — *Computers, Materials & Continua*, 85(2), 2025. Uses a Ryu controller with an ensemble of CNN, LSTM, FFCNN and RNN models, with SMOTE for class imbalance, on a 14-host/4-switch smart-city topology. Detection-only — it does not close the loop with automated blocking, which is exactly the gap this project fills.
3. **DDoS Attacks Detection in SDN Through Network Traffic Feature Selection and Machine Learning Models** — *Telecom (MDPI)*, 6(3):69, 2025. Mininet + OpenDaylight testbed with real attack tools (GoldenEye, Slowloris, HULK, Slowhttptest, XerXes) and CICFlowMeter feature extraction. The authors note their benign traffic is unrealistic (only two generating sources) — this project avoids that by using an established benign-traffic dataset (CICIDS2017) instead of hand-generating it.
4. **Kumar, C. and Ansari, M.S.A. — An Explainable Nature-Inspired Cyber Attack Detection System in Software-Defined IoT Applications** — *Expert Systems with Applications*, 250:123853, 2024. Metaheuristic feature selection + explainable AI, static feature set, detection only, no real-time response.
5. **Kumar, P., Jolfaei, A. and Islam, A.K.M.N. — An Enhanced Deep-Learning Empowered Threat-Hunting Framework for Software-Defined IoT** — *Computers & Security*, 148:104109, 2025. Deep learning for proactive threat hunting, but offline/analyst-driven rather than an inline, automated blocking system.
6. **Toony, A.A. et al. — Multi-Block: A Novel ML-Based Intrusion Detection Framework for SDN-Enabled IoT Networks Using a New Pyramidal Structure** — *Internet of Things*, 26:101231, 2024. Multi-stage "pyramidal" classifier cascade, evaluated on accuracy only — no controller CPU load or detection latency measured. This project reports per-flow inference latency (see §5) as a differentiator.
7. **Federated Learning Framework for IoT Intrusion Detection Using TabTransformer and Nature-Inspired Hyperparameter Optimization** — *Frontiers in Big Data*, 2025. Federated TabTransformer tuned via Electric-Eel-Foraging-Optimization, 99.02% accuracy / 98.87% F1 on N-BaIoT. Not SDN-integrated, no mitigation.
8. **Computationally Efficient Deep Federated Learning with Optimized Feature Selection for IoT Botnet Attack Detection** — *Intelligent Systems with Applications*, 25:200475, 2025. Lightweight per-client models via FedAvg, privacy- and compute-conscious, but ignores SDN control-plane deployment entirely.
9. **Towards Privacy-Preserving Deep Learning for Intelligent IoT Botnet Detection** — *Applied Sciences*, 16(3):1665, 2026. Benchmarks Random Forest against centralised/federated DNN, LSTM and BiLSTM on Bot-IoT and N-BaIoT — used here as a baseline reference for this project's own model evaluation (§5).
10. **A Review of DDoS Attack Detection and Defense Technologies in SDN** — *ACM NISCOMP*, 2025. Survey covering entropy-based detection, self-organising maps, deep-RL multi-layer defences, SDN-Defend, and CNN+SVM hybrids — the related-work backbone and a source of comparison baselines.

## 2. Problem Statement

IoT devices connected through software-defined networks are increasingly conscripted into DDoS and botnet attacks. Traditional defences rely on static, manually written rules that cannot adapt to changing attack patterns, while most existing machine-learning-based detectors stop at producing a classification label — they do not automatically act on that label to block the malicious traffic. This leaves a practical gap between **detecting** an attack and actually **stopping** it. This project addresses that gap directly: a system that classifies IoT network flows as normal or malicious using flow statistics collected by the SDN controller, and automatically installs a block rule at the switch when an attack is detected — closing the loop between detection and defence.

## 3. Objectives & How Each Was Met

| # | Objective | Status | Where |
|---|---|---|---|
| 1 | Dataset preparation: CICIDS2017 (primary) + BoT-IoT/N-BaIoT (secondary), cleaned, normalised, compact feature subset | **Done** | `data/data_prep.py` — auto-maps any of the three datasets' column schemas onto a 19-feature subset; ships with a schema-identical synthetic generator for offline development |
| 2 | Train a lightweight detector: Random Forest / XGBoost primary, DL stretch goal optional | **Done** | `models/train_model.py` — trains and compares both RF and XGBoost, keeps the better one by F1 |
| 3 | Minimal Mininet (4–6 hosts) + Ryu testbed | **Done** | `mininet_topo/topology.py` — 5 hosts (4 normal IoT devices + 1 attacker), 1 OVS switch, remote Ryu controller |
| 4 | Integrate detection with automated mitigation | **Done** | `controller/ryu_mitigation_app.py` — polls `OFPFlowStatsRequest` every 2s, scores each flow, installs a drop rule on detection |

## 4. System Architecture

```
            ┌───────────────────────────┐
 CICIDS2017 │   data_prep.py            │
 /BoT-IoT   │   clean, select 19 feats  │
 (offline)  └──────────────┬────────────┘
                            ▼
                 ┌─────────────────────┐
                 │  train_model.py      │   RandomForest vs XGBoost
                 │  -> model.joblib     │   (best kept by F1 score)
                 └──────────┬───────────┘
                            │
                            ▼
      ┌─────────────────────────────────────────┐
      │  shared/mitigation_logic.py               │
      │  flow stats -> feature vector -> verdict  │◄──────────┐
      └──────────────┬─────────────────┬──────────┘           │
                      ▼                 ▼                      │
     controller/ryu_mitigation_app.py  demo/live_ovs_demo.py   │
     (standard Ryu OF1.3 app)          (ovs-ofctl, no Ryu)     │
                      │                 │                      │
                      ▼                 ▼                      │
           ┌─────────────────────────────────────┐             │
           │   Mininet: iot1-4, atk1, s1 (OVS)     │────────────┘
           │   (mininet_topo/topology.py)          │  periodic flow-stats poll
           └─────────────────────────────────────┘
```

Every stage after training reads the same `model.joblib` / `scaler.joblib`
through `shared/mitigation_logic.py`, so there is exactly one trained
model behind both the offline evaluation and the live controller.

## 5. Methodology

### 5.1 Dataset & features
Eighteen flow-level statistics plus the binary label, selected because a
controller can realistically obtain (or approximate) all of them from
standard OpenFlow counters: flow duration, forward/backward packet and
byte counts, per-packet length mean/std, flow byte/packet rate, inter-
arrival-time mean/std (flow-, forward- and backward-direction), SYN/ACK
flag counts, down/up ratio, and average packet size. `data_prep.py`
accepts either a real CICIDS2017/BoT-IoT/N-BaIoT CSV (auto-mapping
CICFlowMeter's column names) or generates a schema-identical synthetic
dataset modelled on a volumetric SYN/ICMP-flood attack pattern (many
small packets, very short inter-arrival time, high SYN count, almost no
response traffic) versus ordinary bursty IoT traffic.

### 5.2 Model training
Both a `RandomForestClassifier` (200 trees, class-balanced) and an
`XGBClassifier` (300 rounds) are trained on an 80/20 stratified
train/test split, after `StandardScaler` normalisation. Results in §6
are from the dataset actually used for this run — see
`models/artifacts/metrics.json` for exact numbers.

### 5.3 Live mitigation under partial observability
A live `OFPFlowStatsReply` only exposes duration, packet count and byte
count per flow — not the full CICFlowMeter feature set used for offline
training. Rather than approximate the missing features at inference
time (which was tried and under-detects, since most of a full model's
decision weight sits on exactly the features that would have to be
guessed), `train_model.py` trains a **second, smaller model directly on
the 6 features a flow-stats poll can supply** (duration, forward
packet/byte counts, derived byte-rate/packet-rate, average packet size).
This "heavy model offline, light model inline" split is standard
ML-systems practice: `models/artifacts/full/` holds the thorough
19-feature evaluation (Objective 2), `models/artifacts/live/` holds the
model `shared/mitigation_logic.py` actually loads for the controller and
the OVS demo (Objective 4).

### 5.4 Mitigation action
On detection, the controller installs an OpenFlow rule matching the
flagged flow's exact fields with an empty action list (drop) at priority
100 — above the learning-switch's normal priority-1 forwarding rules —
with a 30s idle timeout so transient false positives self-clear.

## 6. Results

Results are on the real CICIDS2017 Friday-afternoon DDoS capture
(214,605 cleaned flows; 80/20 stratified split, 42,920 test flows).

| Model | Features | Algorithm | Accuracy | Precision | Recall | F1 | Inference |
|---|---|---|---|---|---|---|---|
| Full (offline eval) | 19 | RandomForest | 0.9994 | 0.9999 | 0.9991 | 0.9995 | 0.0074 ms/flow |
| Live (inline controller) | 6 | RandomForest | 0.9992 | 0.9998 | 0.9988 | 0.9993 | 0.0112 ms/flow |

**Detect-and-block replay.** Held-out flows scored with the live model at
the 0.70 threshold: 25,556 of 25,602 attack flows flagged (99.82%) and 2
of 17,318 benign flows flagged (0.012%).

**Caveats.** Results are in-distribution (one day, one dataset, random
split). XGBoost was not run on the real data. A synthetic tiny-packet
`hping3`-style SYN flood was *not* flagged by the model trained on real
CICIDS2017 data, because those DDoS flows are short with ~900-byte
packets (domain mismatch).

**Testbed status.** `demo/simulate_demo.py` exercises the decision logic
in pure Python. The Mininet, Ryu and OVS components were implemented but
**not executed** (no Linux VM was available), so installation of a drop
rule on a real switch is unverified.

## 7. Comparison with Surveyed Work

| Work | Detects | Auto-mitigates | Reports latency |
|---|---|---|---|
| [2] RNN-RYU | ✓ | ✗ | ✗ |
| [4] Explainable nature-inspired | ✓ | ✗ | ✗ |
| [5] DL threat-hunting | ✓ (offline) | ✗ | ✗ |
| [6] Multi-Block pyramidal | ✓ | ✗ | ✗ (accuracy-only) |
| **This project** | ✓ | ✓ (OpenFlow drop rule) | ✓ (per-flow ms, see §6) |

## 8. Tools & Datasets Used

| Category | Choice | Why |
|---|---|---|
| SDN Controller | Ryu | Simplest Python controller to script against |
| Emulator | Mininet | Standard, lightweight, quick to script a small topology |
| ML library | scikit-learn (RandomForest) / XGBoost | Fast CPU training, strong baseline on flow data |
| Primary dataset | CICIDS2017 | Large, well-labelled, widely used |
| Secondary dataset | BoT-IoT / N-BaIoT | IoT-specific validation |
| Attack generation | hping3 | Single lightweight tool, enough for one convincing live attack |

## 9. Conclusion & Future Work

This project implements the full detect-and-mitigate loop that most of
the surveyed literature stops short of: a lightweight classifier trained
on CICFlowMeter-style flow statistics, deployed inline in a Ryu
OpenFlow 1.3 controller that polls real flow stats and installs block
rules automatically. Future work (flagged, not required for this
submission): rate-limiting (meter bands) as a softer response than a
hard drop for borderline-confidence flows; a small MLP/1D-CNN stretch
comparison against the tree-based baselines (Objective 2's optional
stretch goal); and validating against a second, IoT-specific dataset
(BoT-IoT / N-BaIoT) alongside CICIDS2017 for cross-dataset
generalisation.
