"""
shared/mitigation_logic.py
---------------------------
Single source of truth for "take flow statistics -> decide benign/attack"
(Objective 4). Both the Ryu controller (controller/ryu_mitigation_app.py)
and the OVS-only live demo (demo/live_ovs_demo.py) import this module, so
the exact same trained model and the exact same decision logic drive both
of them, as well as demo/simulate_demo.py.

Why a dedicated "live" model instead of the full CICFlowMeter one?
--------------------------------------------------------------------
The offline model (models/train_model.py's "full" model) is trained on 19
CICFlowMeter-style features (Objective 1/2). A real SDN data plane can't
hand the controller all 19 of those at line rate on every poll -- an
OpenFlow FlowStatsReply (or `ovs-ofctl dump-flows`) only gives you, per
flow: packet_count, byte_count and duration. Everything else CICFlowMeter
computes (per-packet length mean/std, inter-arrival-time mean/std, TCP
flag counts, ...) needs deep packet inspection of every packet -- too
slow for inline mitigation.

The first version of this module tried to reuse the full 19-feature model
live by filling in the unobservable features with training-set averages.
That backfires: most of the full model's decision weight sits on exactly
the features that have to be guessed at inference time, so the imputed
("probably benign") values drown out the real signal in the handful of
features that ARE observed, and attacks stop getting flagged.

The fix is the standard one: train_model.py also trains a second,
smaller model directly on only the features a live poll can supply
(LIVE_COMPUTABLE below) -- a "heavy model offline, light model inline"
split that is completely ordinary ML-systems practice. That live model is
what this module loads and what both the Ryu app and the OVS demo use.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "artifacts", "live")

# Must match models/train_model.py's LIVE_FEATURES, in the same order the
# model was trained on (feature_list.json, loaded below, is authoritative).
LIVE_COMPUTABLE = [
    "flow_duration", "tot_fwd_pkts", "totlen_fwd_pkts",
    "flow_bytes_per_s", "flow_pkts_per_s", "avg_pkt_size",
]

BLOCK_SCORE_THRESHOLD = 0.70   # predict_proba(attack) above this -> block
MIN_PACKETS_TO_JUDGE = 8       # don't judge a flow until it has sent a few packets


class MitigationEngine:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.model = joblib.load(os.path.join(artifact_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(artifact_dir, "scaler.joblib"))
        with open(os.path.join(artifact_dir, "feature_list.json")) as f:
            self.features = json.load(f)["features"]

    def build_feature_vector(self, duration_s: float, packet_count: int, byte_count: int):
        """Build the feature vector straight from the three numbers a
        flow-stats poll actually gives us -- no imputation needed, because
        the live model was trained on exactly this feature set."""
        pkts = max(packet_count, 0)
        byts = max(byte_count, 0)
        duration_us = max(duration_s, 1e-6) * 1e6

        live = {
            "flow_duration": duration_us,
            "tot_fwd_pkts": pkts,
            "totlen_fwd_pkts": byts,
            "flow_bytes_per_s": byts / max(duration_s, 1e-3),
            "flow_pkts_per_s": pkts / max(duration_s, 1e-3),
            "avg_pkt_size": byts / max(pkts, 1),
        }
        row = {feat: live[feat] for feat in self.features}
        return pd.DataFrame([row], columns=self.features), live

    def predict(self, duration_s: float, packet_count: int, byte_count: int):
        """Returns dict(is_attack, confidence, live_features)."""
        if packet_count < MIN_PACKETS_TO_JUDGE:
            return {"is_attack": False, "confidence": 0.0, "reason": "too_few_packets",
                    "live_features": {}}

        x_raw, live = self.build_feature_vector(duration_s, packet_count, byte_count)
        x_scaled = self.scaler.transform(x_raw)

        if hasattr(self.model, "predict_proba"):
            proba_attack = float(self.model.predict_proba(x_scaled)[0, 1])
        else:
            proba_attack = float(self.model.predict(x_scaled)[0])

        is_attack = proba_attack >= BLOCK_SCORE_THRESHOLD
        return {"is_attack": is_attack, "confidence": proba_attack,
                "reason": "model", "live_features": live}
