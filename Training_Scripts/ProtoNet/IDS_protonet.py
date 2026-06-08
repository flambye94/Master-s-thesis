"""
IDS_protonet.py
===============
ProtoNet for realistic IDS evaluation:
  Each episode = 3 fixed benign classes + 1 random novel attack class (4-way).

This mirrors a real IDS deployment scenario where the model must distinguish
normal traffic from a previously unseen attack type using only K labelled
examples of the new attack.

Key metrics reported:
  - Episode accuracy (overall)
  - Attack Detection Rate (ADR): fraction of attack queries correctly classified
  - False Alarm Rate (FAR): fraction of benign queries misclassified as attack
  - Per-attack-class ADR across all novel attack classes

Training uses the standard 5-way episodic setup on base classes
(benign classes are included in base — model sees benign during training).
Evaluation uses the 4-way fixed benign + 1 novel attack setup.

Usage:
  python IDS_protonet.py [--k_shot 5] [--q_query 15]
                         [--episodes 10000] [--eval_interval 500]
                         [--lr 0.001] [--device cpu]

Novel attack pool — all attack classes NOT in the base training set.
Edit NOVEL_ATTACK_CLASSES below to change the pool.
"""

import os, argparse, random, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_DIR   = r"G:\iot_dataset\balanced_dataset"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── The 3 benign classes — always in support set during evaluation ─
BENIGN_CLASSES = {
    "Benign",
    "Benign_Grafana",
    "Benign_SSH",
}

# ── Novel attack pool — held out from meta-training ───────────────
# These are the attack classes the model has NEVER seen during training.
# During evaluation, one is randomly sampled per episode.
# Edit this set to change the novel attack pool.
NOVEL_ATTACK_CLASSES = {
    "DDoS_UDP_Fragmentation",
    "DDoS_ICMP_Flood",
    "DoS_RSTFIN_Flood",
    "BruteForce_HTTP",
    "Mirai_Greeth_Flood",
    "Recon_Masscan",
    "Spoofing_SSH",
    "Web_XSS",
    "Web_SQLi",
    "FalseDataInjection_Replay",
}

DROP_COLS = {
    "timestamp", "src_ip", "src_port",
    "dst_ip", "service", "label"
}
# NOTE: dst_port is NOT in DROP_COLS — retained as feature.

LOG_COLS = [
    "IAT", "fwd_iat_mean", "fwd_iat_max", "fwd_iat_min", "fwd_iat_std",
    "bwd_iat_mean", "bwd_iat_max", "bwd_iat_min", "bwd_iat_std",
    "flow_duration", "Duration", "Rate", "Srate", "Drate",
    "flow_bytes_s", "flow_packets_s", "Tot_size", "Tot_sum",
    "fwd_packet_length_max", "bwd_packet_length_max",
    "fwd_packet_length_std", "bwd_packet_length_std",
    "total_length_fwd_packets", "total_length_bwd_packets",
    "Variance", "Covariance", "Magnitude",
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k_shot",        type=int,   default=5)
    p.add_argument("--q_query",       type=int,   default=15)
    p.add_argument("--lr",            type=float, default=0.001)
    p.add_argument("--episodes",      type=int,   default=10000)
    p.add_argument("--eval_interval", type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=200)
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--seed",          type=int,   default=42)
    # Training n_way — standard 5-way on base classes during training
    p.add_argument("--train_n_way",   type=int,   default=5)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
def add_features(df):
    eps = 1e-9
    c   = set(df.columns)
    if "fwd_iat_std" in c and "fwd_iat_mean" in c:
        df["fwd_iat_cv"]         = df["fwd_iat_std"] / (df["fwd_iat_mean"] + eps)
    if "bwd_iat_std" in c and "bwd_iat_mean" in c:
        df["bwd_iat_cv"]         = df["bwd_iat_std"] / (df["bwd_iat_mean"] + eps)
    if "fwd_iat_mean" in c and "bwd_iat_mean" in c:
        df["fwd_bwd_iat_ratio"]  = df["fwd_iat_mean"] / (df["bwd_iat_mean"] + eps)
    if "fwd_iat_max" in c and "fwd_iat_min" in c:
        df["fwd_iat_range"]      = df["fwd_iat_max"] - df["fwd_iat_min"]
    if "bwd_iat_max" in c and "bwd_iat_min" in c:
        df["bwd_iat_range"]      = df["bwd_iat_max"] - df["bwd_iat_min"]
    if "IAT" in c and "flow_duration" in c:
        df["iat_duration_ratio"] = df["IAT"] / (df["flow_duration"] + eps)
    if "fwd_packet_length_max" in c and "fwd_packet_length_min" in c:
        df["fwd_pkt_range"]      = df["fwd_packet_length_max"] - df["fwd_packet_length_min"]
    if "bwd_packet_length_max" in c and "bwd_packet_length_min" in c:
        df["bwd_pkt_range"]      = df["bwd_packet_length_max"] - df["bwd_packet_length_min"]
    if "fwd_packet_length_std" in c and "fwd_packet_length_mean" in c:
        df["fwd_pkt_cv"]         = df["fwd_packet_length_std"] / (df["fwd_packet_length_mean"] + eps)
    if "bwd_packet_length_std" in c and "bwd_packet_length_mean" in c:
        df["bwd_pkt_cv"]         = df["bwd_packet_length_std"] / (df["bwd_packet_length_mean"] + eps)
    if "fwd_packet_length_mean" in c and "bwd_packet_length_mean" in c:
        df["pkt_len_asymmetry"]  = (df["fwd_packet_length_mean"] - df["bwd_packet_length_mean"]) / \
                                    (df["fwd_packet_length_mean"] + df["bwd_packet_length_mean"] + eps)
    if "total_length_fwd_packets" in c and "total_length_bwd_packets" in c:
        total = df["total_length_fwd_packets"] + df["total_length_bwd_packets"]
        df["payload_total"]      = total
        df["payload_asymmetry"]  = (df["total_length_fwd_packets"] - df["total_length_bwd_packets"]) / \
                                    (total + eps)
    if "total_fwd_packets" in c and "total_bwd_packets" in c:
        df["pkt_ratio"]          = df["total_fwd_packets"] / (df["total_bwd_packets"] + eps)
        df["total_packets"]      = df["total_fwd_packets"] + df["total_bwd_packets"]
    if "Rate" in c and "Drate" in c:
        df["rate_ratio"]         = df["Rate"] / (df["Drate"] + eps)
        df["rate_asymmetry"]     = (df["Rate"] - df["Drate"]) / (df["Rate"] + df["Drate"] + eps)
    if "Srate" in c and "Drate" in c:
        df["srate_drate_ratio"]  = df["Srate"] / (df["Drate"] + eps)
    if "Header_Length" in c and "Tot_size" in c:
        df["header_ratio"]       = df["Header_Length"] / (df["Tot_size"] + eps)
        df["payload_size"]       = (df["Tot_size"] - df["Header_Length"]).clip(lower=0)
    flag_cols = [f for f in ["fin_flag_number", "syn_flag_number",
                              "psh_flag_number", "rst_flag_number"] if f in c]
    if flag_cols:
        df["flag_sum"]           = df[flag_cols].sum(axis=1)
    if "syn_flag_number" in c and "fin_flag_number" in c:
        df["syn_fin_ratio"]      = df["syn_flag_number"] / (df["fin_flag_number"] + eps)
    if "psh_flag_number" in c and "syn_flag_number" in c:
        df["psh_syn_ratio"]      = df["psh_flag_number"] / (df["syn_flag_number"] + eps)
    if "Max" in c and "Min" in c:
        df["stat_range"]         = df["Max"] - df["Min"]
    if "Std" in c and "AVG" in c:
        df["stat_cv"]            = df["Std"] / (df["AVG"] + eps)
    if "Variance" in c and "AVG" in c:
        df["variance_mean_ratio"]= df["Variance"] / (df["AVG"] + eps)
    if "flow_bytes_s" in c and "flow_packets_s" in c:
        df["bytes_per_packet"]   = df["flow_bytes_s"] / (df["flow_packets_s"] + eps)
    return df


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_and_preprocess(data_dir):
    print("Loading dataset...")
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"),
                             recursive=True))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "label" in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")

    data = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(data):,} rows, "
          f"{len(data['label'].unique())} classes")

    labels    = data["label"].values
    feat_cols = [c for c in data.columns if c not in DROP_COLS]
    X = data[feat_cols].copy()

    for col in LOG_COLS:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    X = add_features(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    std   = X.std()
    const = std[std == 0].index.tolist()
    if const:
        print(f"  Dropping {len(const)} constant features")
        X = X.drop(columns=const)

    feat_names = list(X.columns)
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X.astype(np.float64)).astype(np.float32)

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    print(f"  Features: {len(feat_names)}  Classes: {len(le.classes_)}")
    return X_scaled, y, le, feat_names


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
class MLPEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# STANDARD EPISODE SAMPLER (for meta-training on base classes)
# ─────────────────────────────────────────────────────────────────
class EpisodeSampler:
    def __init__(self, data_x, data_y, class_list,
                 n_way, k_shot, q_query, device):
        self.data_x  = torch.tensor(data_x,
                       dtype=torch.float32).to(device)
        self.data_y  = data_y
        self.n_way   = n_way
        self.k_shot  = k_shot
        self.q_query = q_query
        self.device  = device
        self.class_idx = {}
        for cls in class_list:
            idx = np.where(data_y == cls)[0]
            if len(idx) >= k_shot + q_query:
                self.class_idx[cls] = idx
        self.valid_classes = list(self.class_idx.keys())
        assert len(self.valid_classes) >= n_way, \
            f"Not enough classes: {len(self.valid_classes)} < {n_way}"

    def sample_episode(self):
        chosen = random.sample(self.valid_classes, self.n_way)
        support_x, support_y, query_x, query_y = [], [], [], []
        for local_label, cls in enumerate(chosen):
            idx  = self.class_idx[cls]
            perm = np.random.permutation(len(idx))
            sup_idx = idx[perm[:self.k_shot]]
            qry_idx = idx[perm[self.k_shot:self.k_shot + self.q_query]]
            support_x.append(self.data_x[sup_idx])
            support_y.extend([local_label] * self.k_shot)
            query_x.append(self.data_x[qry_idx])
            query_y.extend([local_label] * self.q_query)
        return (
            torch.cat(support_x),
            torch.tensor(support_y, dtype=torch.long).to(self.device),
            torch.cat(query_x),
            torch.tensor(query_y, dtype=torch.long).to(self.device)
        )


# ─────────────────────────────────────────────────────────────────
# REAL IDS SAMPLER
# 4-way: local labels 0,1,2 = benign types, 3 = novel attack
# ─────────────────────────────────────────────────────────────────
class RealIDSSampler:
    def __init__(self, data_x, data_y, benign_indices,
                 novel_attack_indices, k_shot, q_query, device, le):
        self.data_x  = torch.tensor(data_x,
                       dtype=torch.float32).to(device)
        self.data_y  = data_y
        self.k_shot  = k_shot
        self.q_query = q_query
        self.device  = device
        self.n_way   = 4  # fixed: 3 benign + 1 attack
        self.le      = le  # LabelEncoder for resolving names

        # Index benign classes — keyed by integer index
        self.benign_idx = {}
        for cls in benign_indices:
            idx = np.where(data_y == cls)[0]
            if len(idx) >= k_shot + q_query:
                self.benign_idx[cls] = idx
        assert len(self.benign_idx) == 3, \
            f"Expected 3 benign classes, got {len(self.benign_idx)}"

        # Index novel attack classes — keyed by CLASS NAME string
        self.attack_idx = {}
        for cls_int in novel_attack_indices:
            idx = np.where(data_y == cls_int)[0]
            if len(idx) >= k_shot + q_query:
                cls_name = le.classes_[cls_int]  # resolve to name
                self.attack_idx[cls_name] = idx
        self.valid_attacks = list(self.attack_idx.keys())
        print(f"  RealIDS sampler: 3 benign classes + "
              f"{len(self.valid_attacks)} novel attack classes")

    def sample_episode(self, attack_cls=None):
        if attack_cls is None:
            attack_cls = random.choice(self.valid_attacks)
        # attack_cls is now always a class name string

        # local labels: 0,1,2 = benign types, 3 = attack
        chosen_benign = list(self.benign_idx.keys())
        support_x, support_y, query_x, query_y = [], [], [], []

        # Add 3 benign classes (local labels 0,1,2)
        for local_label, cls in enumerate(chosen_benign):
            idx  = self.benign_idx[cls]
            perm = np.random.permutation(len(idx))
            sup_idx = idx[perm[:self.k_shot]]
            qry_idx = idx[perm[self.k_shot:self.k_shot + self.q_query]]
            support_x.append(self.data_x[sup_idx])
            support_y.extend([local_label] * self.k_shot)
            query_x.append(self.data_x[qry_idx])
            query_y.extend([local_label] * self.q_query)

        # Add 1 novel attack class (local label 3)
        idx  = self.attack_idx[attack_cls]
        perm = np.random.permutation(len(idx))
        sup_idx = idx[perm[:self.k_shot]]
        qry_idx = idx[perm[self.k_shot:self.k_shot + self.q_query]]
        support_x.append(self.data_x[sup_idx])
        support_y.extend([3] * self.k_shot)
        query_x.append(self.data_x[qry_idx])
        query_y.extend([3] * self.q_query)

        return (
            torch.cat(support_x),
            torch.tensor(support_y, dtype=torch.long).to(self.device),
            torch.cat(query_x),
            torch.tensor(query_y, dtype=torch.long).to(self.device),
            attack_cls  # always a name string
        )


# ─────────────────────────────────────────────────────────────────
# PROTONET LOSS
# ─────────────────────────────────────────────────────────────────
def prototypical_loss(support_emb, support_y, query_emb, query_y, n_way):
    prototypes = torch.stack([
        support_emb[support_y == c].mean(0) for c in range(n_way)
    ])
    dists  = torch.cdist(query_emb, prototypes)
    log_p  = F.log_softmax(-dists, dim=1)
    loss   = F.nll_loss(log_p, query_y)
    acc    = (log_p.argmax(dim=1) == query_y).float().mean().item()
    return loss, acc, log_p.argmax(dim=1)


# ─────────────────────────────────────────────────────────────────
# STANDARD EVALUATE (for training checkpoints on base classes)
# ─────────────────────────────────────────────────────────────────
def evaluate(model, sampler, n_episodes, device):
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            sup_x, sup_y, qry_x, qry_y = sampler.sample_episode()
            _, acc, _ = prototypical_loss(
                model(sup_x), sup_y, model(qry_x), qry_y, sampler.n_way)
            accs.append(acc)
    model.train()
    mean_acc = np.mean(accs)
    ci95     = 1.96 * np.std(accs) / np.sqrt(n_episodes)
    return mean_acc, ci95


# ─────────────────────────────────────────────────────────────────
# REAL IDS EVALUATE
# Returns: overall acc, ADR, FAR, per-attack ADR
# ADR = Attack Detection Rate = recall on attack class (local label 3)
# FAR = False Alarm Rate = fraction of benign queries misclassified
# ─────────────────────────────────────────────────────────────────
def evaluate_ids(model, ids_sampler, n_episodes, device):
    model.eval()
    accs         = []
    adr_list     = []   # per episode attack detection rate
    far_list     = []   # per episode false alarm rate
    # per attack class: [correct_detections, total_queries]
    per_class_adr = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for _ in range(n_episodes):
            sup_x, sup_y, qry_x, qry_y, atk_cls = \
                ids_sampler.sample_episode()
            _, acc, preds = prototypical_loss(
                model(sup_x), sup_y,
                model(qry_x), qry_y, ids_sampler.n_way)
            accs.append(acc)

            # local label 3 = attack class
            attack_mask = (qry_y == 3)
            benign_mask = (qry_y != 3)

            # ADR: fraction of attack queries correctly predicted as 3
            # atk_cls is already the class name string from sample_episode()
            if attack_mask.sum() > 0:
                adr = (preds[attack_mask] == 3).float().mean().item()
                adr_list.append(adr)
                # Use atk_cls directly — it is the class name string
                per_class_adr[atk_cls][0] += \
                    (preds[attack_mask] == 3).sum().item()
                per_class_adr[atk_cls][1] += attack_mask.sum().item()

            # FAR: fraction of benign queries predicted as attack (label 3)
            if benign_mask.sum() > 0:
                far = (preds[benign_mask] == 3).float().mean().item()
                far_list.append(far)

    model.train()
    mean_acc = np.mean(accs)
    ci_acc   = 1.96 * np.std(accs) / np.sqrt(n_episodes)
    mean_adr = np.mean(adr_list) if adr_list else 0.0
    ci_adr   = 1.96 * np.std(adr_list) / np.sqrt(len(adr_list)) \
               if len(adr_list) > 1 else 0.0
    mean_far = np.mean(far_list) if far_list else 0.0
    ci_far   = 1.96 * np.std(far_list) / np.sqrt(len(far_list)) \
               if len(far_list) > 1 else 0.0

    # Per-class ADR
    per_class = {}
    for cls, (correct, total) in per_class_adr.items():
        per_class[cls] = correct / total if total > 0 else 0.0

    return mean_acc, ci_acc, mean_adr, ci_adr, mean_far, ci_far, per_class


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 72)
    print(f"IDS ProtoNet  |  4-way (3 benign + 1 novel attack)  |  "
          f"{args.k_shot}-shot  |  device={args.device}")
    print("=" * 72)

    X, y, le, feat_names = load_and_preprocess(DATA_DIR)
    in_dim = len(feat_names)

    with open(os.path.join(OUTPUT_DIR,
              "IDS_protonet_feature_names.json"), "w") as fh:
        json.dump(feat_names, fh)

    # ── Resolve class indices ─────────────────────────────────────
    all_classes = list(le.classes_)

    benign_names = [c for c in all_classes if c in BENIGN_CLASSES]
    novel_attack_names = [c for c in all_classes
                          if c in NOVEL_ATTACK_CLASSES]
    missing_benign = BENIGN_CLASSES - set(benign_names)
    missing_novel  = NOVEL_ATTACK_CLASSES - set(novel_attack_names)
    if missing_benign:
        print(f"  WARNING: missing benign classes: {missing_benign}")
    if missing_novel:
        print(f"  WARNING: missing novel attacks: {missing_novel}")

    benign_idx         = list(le.transform(benign_names))
    novel_attack_idx   = list(le.transform(novel_attack_names))

    # Base classes = all classes except novel attacks
    # Benign classes ARE in base — model sees benign during training
    base_names = [c for c in all_classes
                  if c not in NOVEL_ATTACK_CLASSES]
    base_idx   = list(le.transform(base_names))

    print(f"\n  Benign classes  ({len(benign_names)}): {benign_names}")
    print(f"  Novel attacks   ({len(novel_attack_names)}): "
          f"{novel_attack_names}")
    print(f"  Base classes    ({len(base_names)}): "
          f"all classes except novel attacks")
    print(f"  Input dim: {in_dim}")

    # ── Data splits ───────────────────────────────────────────────
    base_mask        = np.isin(y, base_idx)
    novel_attack_mask= np.isin(y, novel_attack_idx)
    benign_mask      = np.isin(y, benign_idx)

    base_x,  base_y  = X[base_mask],  y[base_mask]
    ids_x = np.concatenate([X[benign_mask], X[novel_attack_mask]])
    ids_y = np.concatenate([y[benign_mask], y[novel_attack_mask]])

    # ── Samplers ─────────────────────────────────────────────────
    train_sampler = EpisodeSampler(
        base_x, base_y, base_idx,
        args.train_n_way, args.k_shot, args.q_query, device)

    # Pass integer indices — RealIDSSampler will use le to resolve names
    ids_sampler = RealIDSSampler(
        ids_x, ids_y, benign_idx, novel_attack_idx,
        k_shot=args.k_shot, q_query=args.q_query,
        device=device, le=le)

    print(f"\n  Train sampler: {len(train_sampler.valid_classes)} "
          f"valid base classes ({args.train_n_way}-way)")

    # ── Model ─────────────────────────────────────────────────────
    model = MLPEmbedding(in_dim).to(device)
    opt   = Adam(model.parameters(), lr=args.lr)
    sched = StepLR(opt, step_size=2000, gamma=0.5)

    print(f"\n  Model params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"  Episodes: {args.episodes}  LR: {args.lr}")

    print("\n" + "=" * 72)
    print("META-TRAINING (standard 5-way on base classes)")
    print("=" * 72)

    best_adr      = 0.0
    history       = defaultdict(list)
    best_pt       = os.path.join(OUTPUT_DIR, "IDS_protonet_best.pt")

    for ep in range(1, args.episodes + 1):
        model.train()
        sup_x, sup_y, qry_x, qry_y = train_sampler.sample_episode()
        loss, acc, _ = prototypical_loss(
            model(sup_x), sup_y,
            model(qry_x), qry_y, args.train_n_way)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc)

        if ep % args.eval_interval == 0:
            # Standard base eval
            base_acc, base_ci = evaluate(
                model, train_sampler, args.eval_episodes, device)

            # IDS eval — 4-way benign vs novel attack
            ep_acc, ep_ci, adr, adr_ci, far, far_ci, _ = evaluate_ids(
                model, ids_sampler, args.eval_episodes, device)

            history["base_acc"].append(base_acc)
            history["ids_acc"].append(ep_acc)
            history["adr"].append(adr)
            history["far"].append(far)

            print(f"  Ep {ep:>6}  "
                  f"loss={np.mean(history['train_loss'][-100:]):.4f}  "
                  f"base_acc={base_acc:.4f}±{base_ci:.4f}  "
                  f"IDS_acc={ep_acc:.4f}±{ep_ci:.4f}  "
                  f"ADR={adr:.4f}±{adr_ci:.4f}  "
                  f"FAR={far:.4f}±{far_ci:.4f}")

            if adr > best_adr:
                best_adr = adr
                torch.save(model.state_dict(), best_pt)

    # ── Final evaluation ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL IDS EVALUATION")
    print("4-way: 3 benign classes + 1 novel attack class")
    print("=" * 72)

    model.load_state_dict(torch.load(best_pt, map_location=device))

    for k in [1, 5, 10]:
        if k > args.k_shot:
            continue
        s = RealIDSSampler(
            ids_x, ids_y, benign_idx, novel_attack_idx,
            k_shot=k, q_query=args.q_query,
            device=device, le=le)
        ep_acc, ep_ci, adr, adr_ci, far, far_ci, per_class = \
            evaluate_ids(model, s, args.eval_episodes * 2, device)

        print(f"\n  4-way {k:>2}-shot:")
        print(f"    Episode Accuracy : {ep_acc:.4f} ± {ep_ci:.4f}")
        print(f"    Attack Det. Rate : {adr:.4f} ± {adr_ci:.4f}  "
              f"(fraction of attack traffic correctly flagged)")
        print(f"    False Alarm Rate : {far:.4f} ± {far_ci:.4f}  "
              f"(fraction of benign traffic falsely flagged)")
        print(f"\n    Per-class Attack Detection Rate ({k}-shot):")
        for cls in sorted(per_class.keys()):
            print(f"      {cls:<40} ADR={per_class[cls]:.4f}")

    print(f"\n  Best ADR during training: {best_adr:.4f}")
    np.save(os.path.join(OUTPUT_DIR,
            "IDS_protonet_history.npy"), dict(history))
    print(f"  Model saved → {best_pt}")
    print("=" * 72)


if __name__ == "__main__":
    main()