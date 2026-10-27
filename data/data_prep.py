"""
data_prep.py
------------
Dataset preparation & preprocessing for the SDN IoT Anomaly Detection project.

Objective 1 from the project proposal:
    "Use CICIDS2017 as the primary source of labelled DDoS/benign traffic,
    with BoT-IoT or N-BaIoT as a secondary IoT-specific dataset for
    validation. Clean, normalise and select a manageable subset of
    features."

USAGE
-----
1) With a real dataset (recommended for the final submission):
   Download CICIDS2017 (e.g. the "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
   file from https://www.unb.ca/cic/datasets/ids-2017.html) or a BoT-IoT /
   N-BaIoT CSV, then run:

       python3 data_prep.py --csv /path/to/CICIDS2017.csv --out ./data/processed

   The script auto-detects the label column and common CICFlowMeter column
   names (it also strips the leading spaces CICFlowMeter puts in its
   headers) and works with CICIDS2017, BoT-IoT and N-BaIoT files without
   modification.

2) Without internet access / for a quick dry run (what this repo ships
   with by default), omit --csv and the script generates a synthetic but
   statistically realistic flow dataset using the SAME feature schema as
   CICFlowMeter/CICIDS2017, so the rest of the pipeline (training, the Ryu
   app, the demo) is byte-for-byte identical either way:

       python3 data_prep.py --synthetic --rows 20000 --out ./data/processed

The output is always two files: processed/train.csv and processed/test.csv,
plus processed/feature_list.json describing the selected feature subset.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# The compact, manageable feature subset we train on (Objective 1).
# These are flow-level statistics any OpenFlow/Ryu controller can compute
# from stats it already polls (packet counts, byte counts, duration) plus
# a handful of CICFlowMeter-style derived stats, so the same feature
# vector can be rebuilt live by ryu_mitigation_app.py.
SELECTED_FEATURES = [
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "flow_bytes_per_s",
    "flow_pkts_per_s",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "pkt_len_mean",
    "pkt_len_std",
    "syn_flag_cnt",
    "ack_flag_cnt",
    "down_up_ratio",
    "avg_pkt_size",
]

LABEL_COL = "label"

# Common raw column name variants across CICIDS2017 / BoT-IoT / N-BaIoT CSVs
# that map onto our SELECTED_FEATURES. CICFlowMeter headers also often have
# a leading space, which we strip before matching.
COLUMN_ALIASES = {
    "flow_duration": ["flow duration", "flow_duration", "dur"],
    "tot_fwd_pkts": ["total fwd packets", "tot fwd pkts", "spkts"],
    "tot_bwd_pkts": ["total backward packets", "tot bwd pkts", "dpkts"],
    "totlen_fwd_pkts": ["total length of fwd packets", "totlen fwd pkts", "sbytes"],
    "totlen_bwd_pkts": ["total length of bwd packets", "totlen bwd pkts", "dbytes"],
    "fwd_pkt_len_mean": ["fwd packet length mean", "fwd pkt len mean"],
    "bwd_pkt_len_mean": ["bwd packet length mean", "bwd pkt len mean"],
    "flow_bytes_per_s": ["flow bytes/s", "flow_bytes/s"],
    "flow_pkts_per_s": ["flow packets/s", "flow_pkts/s"],
    "flow_iat_mean": ["flow iat mean"],
    "flow_iat_std": ["flow iat std"],
    "fwd_iat_mean": ["fwd iat mean"],
    "bwd_iat_mean": ["bwd iat mean"],
    "pkt_len_mean": ["packet length mean", "pkt len mean"],
    "pkt_len_std": ["packet length std", "pkt len std"],
    "syn_flag_cnt": ["syn flag count", "syn flag cnt"],
    "ack_flag_cnt": ["ack flag count", "ack flag cnt"],
    "down_up_ratio": ["down/up ratio", "down_up_ratio"],
    "avg_pkt_size": ["average packet size", "avg pkt size"],
}

LABEL_ALIASES = ["label", " label", "attack_cat", "class", "category"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _map_real_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Map an arbitrary CICIDS2017 / BoT-IoT / N-BaIoT CSV onto SELECTED_FEATURES."""
    df = _normalise_columns(df)

    label_col = None
    for alias in LABEL_ALIASES:
        if alias.strip() in df.columns:
            label_col = alias.strip()
            break
    if label_col is None:
        raise ValueError(
            "Could not find a label column. Expected one of: " + ", ".join(LABEL_ALIASES)
        )

    out = pd.DataFrame()
    for feat, aliases in COLUMN_ALIASES.items():
        found = next((a for a in aliases if a in df.columns), None)
        if found is not None:
            out[feat] = pd.to_numeric(df[found], errors="coerce")
        else:
            out[feat] = np.nan  # filled in later

    # Binary label: BENIGN (or "Normal") -> 0, everything else -> 1
    raw_label = df[label_col].astype(str).str.strip().str.lower()
    out[LABEL_COL] = (~raw_label.isin(["benign", "normal", "0"])).astype(int)

    return out


def _generate_synthetic(n_rows: int, attack_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate a synthetic flow dataset with the same column schema as a real
    CICIDS2017/BoT-IoT export. Benign flows look like ordinary IoT traffic
    (short, bursty, small packets). Attack flows are modelled after the
    SYN-flood / ICMP-flood / volumetric-botnet pattern used in Objective 3
    of the project (many tiny packets, very short inter-arrival time, high
    SYN flag count, near-zero response from the victim).
    """
    n_attack = int(n_rows * attack_ratio)
    n_benign = n_rows - n_attack

    def clip_pos(x):
        return np.clip(x, 0, None)

    benign = pd.DataFrame({
        "flow_duration": clip_pos(rng.normal(1_200_000, 600_000, n_benign)),      # microseconds
        "tot_fwd_pkts": clip_pos(rng.poisson(12, n_benign)),
        "tot_bwd_pkts": clip_pos(rng.poisson(10, n_benign)),
        "totlen_fwd_pkts": clip_pos(rng.normal(900, 400, n_benign)),
        "totlen_bwd_pkts": clip_pos(rng.normal(1100, 500, n_benign)),
        "fwd_pkt_len_mean": clip_pos(rng.normal(75, 25, n_benign)),
        "bwd_pkt_len_mean": clip_pos(rng.normal(110, 40, n_benign)),
        "flow_bytes_per_s": clip_pos(rng.normal(2500, 1200, n_benign)),
        "flow_pkts_per_s": clip_pos(rng.normal(18, 8, n_benign)),
        "flow_iat_mean": clip_pos(rng.normal(90_000, 40_000, n_benign)),
        "flow_iat_std": clip_pos(rng.normal(30_000, 15_000, n_benign)),
        "fwd_iat_mean": clip_pos(rng.normal(95_000, 45_000, n_benign)),
        "bwd_iat_mean": clip_pos(rng.normal(100_000, 50_000, n_benign)),
        "pkt_len_mean": clip_pos(rng.normal(90, 30, n_benign)),
        "pkt_len_std": clip_pos(rng.normal(35, 15, n_benign)),
        "syn_flag_cnt": rng.poisson(1, n_benign),
        "ack_flag_cnt": clip_pos(rng.poisson(14, n_benign)),
        "down_up_ratio": clip_pos(rng.normal(1.1, 0.4, n_benign)),
        "avg_pkt_size": clip_pos(rng.normal(95, 30, n_benign)),
    })
    benign[LABEL_COL] = 0

    attack = pd.DataFrame({
        "flow_duration": clip_pos(rng.normal(80_000, 50_000, n_attack)),
        "tot_fwd_pkts": clip_pos(rng.poisson(400, n_attack)),
        "tot_bwd_pkts": clip_pos(rng.poisson(2, n_attack)),
        "totlen_fwd_pkts": clip_pos(rng.normal(24_000, 9_000, n_attack)),
        "totlen_bwd_pkts": clip_pos(rng.normal(60, 50, n_attack)),
        "fwd_pkt_len_mean": clip_pos(rng.normal(60, 10, n_attack)),
        "bwd_pkt_len_mean": clip_pos(rng.normal(20, 15, n_attack)),
        "flow_bytes_per_s": clip_pos(rng.normal(180_000, 60_000, n_attack)),
        "flow_pkts_per_s": clip_pos(rng.normal(900, 300, n_attack)),
        "flow_iat_mean": clip_pos(rng.normal(200, 150, n_attack)),
        "flow_iat_std": clip_pos(rng.normal(120, 90, n_attack)),
        "fwd_iat_mean": clip_pos(rng.normal(180, 140, n_attack)),
        "bwd_iat_mean": clip_pos(rng.normal(5000, 4000, n_attack)),
        "pkt_len_mean": clip_pos(rng.normal(55, 10, n_attack)),
        "pkt_len_std": clip_pos(rng.normal(8, 5, n_attack)),
        "syn_flag_cnt": clip_pos(rng.poisson(350, n_attack)),
        "ack_flag_cnt": clip_pos(rng.poisson(3, n_attack)),
        "down_up_ratio": clip_pos(rng.normal(0.05, 0.05, n_attack)),
        "avg_pkt_size": clip_pos(rng.normal(58, 12, n_attack)),
    })
    attack[LABEL_COL] = 1

    df = pd.concat([benign, attack], ignore_index=True)
    return df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


def clean_and_select(df: pd.DataFrame) -> pd.DataFrame:
    df = df[SELECTED_FEATURES + [LABEL_COL]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in SELECTED_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    # Drop exact duplicate flows (common in CICIDS2017)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str, default=None, help="Path to a real CICIDS2017/BoT-IoT/N-BaIoT CSV")
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic data even if --csv is given")
    ap.add_argument("--rows", type=int, default=20000, help="Rows to generate in synthetic mode")
    ap.add_argument("--attack-ratio", type=float, default=0.30, help="Fraction of attack flows in synthetic mode")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "processed"))
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(args.out, exist_ok=True)

    if args.csv and not args.synthetic:
        print(f"[data_prep] Loading real dataset: {args.csv}")
        raw = pd.read_csv(args.csv, low_memory=False)
        df = _map_real_dataset(raw)
        source = os.path.basename(args.csv)
    else:
        print(f"[data_prep] Generating synthetic CICIDS2017-schema dataset "
              f"({args.rows} rows, {args.attack_ratio:.0%} attack)")
        df = _generate_synthetic(args.rows, args.attack_ratio, rng)
        source = "synthetic"

    df = clean_and_select(df)

    # stratified split
    attack = df[df[LABEL_COL] == 1]
    benign = df[df[LABEL_COL] == 0]
    n_test_a = max(1, int(len(attack) * args.test_size))
    n_test_b = max(1, int(len(benign) * args.test_size))

    test = pd.concat([attack.sample(n_test_a, random_state=RANDOM_SEED),
                       benign.sample(n_test_b, random_state=RANDOM_SEED)])
    train = df.drop(test.index)

    train = train.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    train_path = os.path.join(args.out, "train.csv")
    test_path = os.path.join(args.out, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    with open(os.path.join(args.out, "feature_list.json"), "w") as f:
        json.dump({"features": SELECTED_FEATURES, "label": LABEL_COL, "source": source}, f, indent=2)

    print(f"[data_prep] source={source}")
    print(f"[data_prep] train: {len(train)} rows ({train[LABEL_COL].mean():.1%} attack)")
    print(f"[data_prep] test:  {len(test)} rows ({test[LABEL_COL].mean():.1%} attack)")
    print(f"[data_prep] wrote {train_path}")
    print(f"[data_prep] wrote {test_path}")


if __name__ == "__main__":
    main()
