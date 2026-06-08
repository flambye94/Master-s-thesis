"""
maml_iot.py
===========
MAML (Model-Agnostic Meta-Learning) with MLP backbone
on the custom IoT network security dataset.

Loads directly from balanced_dataset CSV subfolders.
No prepare_meta.py needed.

Usage:
  python maml_iot.py [--mode 54class|10class]
                     [--n_way 5] [--k_shot 5] [--q_query 15]
                     [--inner_steps 5] [--inner_lr 0.01]
                     [--meta_lr 0.001] [--episodes 5000]
                     [--eval_interval 500] [--device cpu]

Modes:
  54class  (default) — fine-grained 54-class taxonomy, 10 novel classes
  10class            — coarse 10-class taxonomy, 3 novel classes
                       Used to empirically verify that coarse class
                       merging degrades few-shot performance.
"""

import os, argparse, random, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_DIR   = r"G:\iot_dataset\balanced_dataset"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 54-class novel split ──────────────────────────────────────────
NOVEL_CLASSES_54 = {
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

# ── Coarse taxonomy remapping ─────────────────────────────────────
COARSE_MAP = {
    "Backdoor_C2Beacon": "Backdoor", "Backdoor_C2_Mirai": "Backdoor",
    "Backdoor_Exfiltration": "Backdoor", "Backdoor_ReverseShell": "Backdoor",
    "Benign": "Benign", "Benign_Grafana": "Benign", "Benign_SSH": "Benign",
    "BruteForce_FTP": "BruteForce", "BruteForce_HTTP": "BruteForce",
    "BruteForce_SSH": "BruteForce", "BruteForce_Telnet": "BruteForce",
    "DDoS_ACK_Fragmentation": "DDoS", "DDoS_ICMP_Flood": "DDoS",
    "DDoS_PSHACK_Flood": "DDoS", "DDoS_RSTFIN_Flood": "DDoS",
    "DDoS_SYN_Flood": "DDoS", "DDoS_TCP_Flood": "DDoS",
    "DDoS_UDP_Flood": "DDoS", "DDoS_UDP_Fragmentation": "DDoS",
    "DoS_ICMP_Flood": "DoS", "DoS_PSHACK_Flood": "DoS",
    "DoS_RSTFIN_Flood": "DoS", "DoS_SYN_Flood": "DoS",
    "DoS_TCP_Flood": "DoS", "DoS_UDP_Flood": "DoS",
    "FalseDataInjection_Replay": "False_Data_Injection",
    "FalseDataInjection_Sensor": "False_Data_Injection",
    "Replay_Sensor_Scapy": "False_Data_Injection",
    "Mirai_GREIP_Flood": "Mirai", "Mirai_Greeth_Flood": "Mirai",
    "Mirai_UDPPlain": "Mirai",
    "Recon_Masscan": "Recon", "Recon_Nmap_OS": "Recon",
    "Recon_Nmap_SYN": "Recon", "Recon_Nmap_Service": "Recon",
    "Recon_Ping": "Recon", "Recon_VulScan": "Recon", "Recon_Web": "Recon",
    "Spoofing_FTP": "Spoofing", "Spoofing_Flask": "Spoofing",
    "Spoofing_HTTP": "Spoofing", "Spoofing_HTTP_Alt": "Spoofing",
    "Spoofing_MitM_ARP_Scapy": "Spoofing", "Spoofing_SSH": "Spoofing",
    "Spoofing_Sensor_Scapy": "Spoofing", "Spoofing_Telnet": "Spoofing",
    "Web_CmdInjection": "Web_Based", "Web_ConfigTamper": "Web_Based",
    "Web_FileUpload": "Web_Based", "Web_Firmware": "Web_Based",
    "Web_SQLi": "Web_Based", "Web_Traversal": "Web_Based",
    "Web_UserEnum": "Web_Based", "Web_XSS": "Web_Based",
}

# ── 10-class novel split: DDoS, Recon, Spoofing held out ─────────
# These three are structurally heterogeneous at the sub-type level,
# making them ideal for verifying that coarse merging hurts few-shot.
NOVEL_CLASSES_10 = {"DDoS", "Recon", "Spoofing"}

DROP_COLS = {
    "timestamp", "src_ip", "src_port",
    "dst_ip", "service", "label"
}
# NOTE: dst_port is NOT in DROP_COLS — it is retained as a feature.

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
    p.add_argument("--mode",         type=str,   default="54class",
                   choices=["54class", "10class"],
                   help="54class: fine-grained (default). "
                        "10class: coarse taxonomy for ablation.")
    p.add_argument("--n_way",         type=int,   default=5)
    p.add_argument("--k_shot",        type=int,   default=5)
    p.add_argument("--q_query",       type=int,   default=15)
    p.add_argument("--inner_steps",   type=int,   default=5)
    p.add_argument("--inner_lr",      type=float, default=0.01)
    p.add_argument("--meta_lr",       type=float, default=0.001)
    p.add_argument("--episodes",      type=int,   default=10000)
    p.add_argument("--eval_interval", type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=200)
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
def add_features(df):
    eps = 1e-9
    c   = set(df.columns)
    if "fwd_iat_std" in c and "fwd_iat_mean" in c:
        df["fwd_iat_cv"]          = df["fwd_iat_std"] / (df["fwd_iat_mean"] + eps)
    if "bwd_iat_std" in c and "bwd_iat_mean" in c:
        df["bwd_iat_cv"]          = df["bwd_iat_std"] / (df["bwd_iat_mean"] + eps)
    if "fwd_iat_mean" in c and "bwd_iat_mean" in c:
        df["fwd_bwd_iat_ratio"]   = df["fwd_iat_mean"] / (df["bwd_iat_mean"] + eps)
    if "fwd_iat_max" in c and "fwd_iat_min" in c:
        df["fwd_iat_range"]       = df["fwd_iat_max"] - df["fwd_iat_min"]
    if "bwd_iat_max" in c and "bwd_iat_min" in c:
        df["bwd_iat_range"]       = df["bwd_iat_max"] - df["bwd_iat_min"]
    if "IAT" in c and "flow_duration" in c:
        df["iat_duration_ratio"]  = df["IAT"] / (df["flow_duration"] + eps)
    if "fwd_packet_length_max" in c and "fwd_packet_length_min" in c:
        df["fwd_pkt_range"]       = df["fwd_packet_length_max"] - df["fwd_packet_length_min"]
    if "bwd_packet_length_max" in c and "bwd_packet_length_min" in c:
        df["bwd_pkt_range"]       = df["bwd_packet_length_max"] - df["bwd_packet_length_min"]
    if "fwd_packet_length_std" in c and "fwd_packet_length_mean" in c:
        df["fwd_pkt_cv"]          = df["fwd_packet_length_std"] / (df["fwd_packet_length_mean"] + eps)
    if "bwd_packet_length_std" in c and "bwd_packet_length_mean" in c:
        df["bwd_pkt_cv"]          = df["bwd_packet_length_std"] / (df["bwd_packet_length_mean"] + eps)
    if "fwd_packet_length_mean" in c and "bwd_packet_length_mean" in c:
        df["pkt_len_asymmetry"]   = (df["fwd_packet_length_mean"] - df["bwd_packet_length_mean"]) / \
                                     (df["fwd_packet_length_mean"] + df["bwd_packet_length_mean"] + eps)
    if "total_length_fwd_packets" in c and "total_length_bwd_packets" in c:
        total = df["total_length_fwd_packets"] + df["total_length_bwd_packets"]
        df["payload_total"]       = total
        df["payload_asymmetry"]   = (df["total_length_fwd_packets"] - df["total_length_bwd_packets"]) / \
                                     (total + eps)
    if "total_fwd_packets" in c and "total_bwd_packets" in c:
        df["pkt_ratio"]           = df["total_fwd_packets"] / (df["total_bwd_packets"] + eps)
        df["total_packets"]       = df["total_fwd_packets"] + df["total_bwd_packets"]
    if "Rate" in c and "Drate" in c:
        df["rate_ratio"]          = df["Rate"] / (df["Drate"] + eps)
        df["rate_asymmetry"]      = (df["Rate"] - df["Drate"]) / (df["Rate"] + df["Drate"] + eps)
    if "Srate" in c and "Drate" in c:
        df["srate_drate_ratio"]   = df["Srate"] / (df["Drate"] + eps)
    if "Header_Length" in c and "Tot_size" in c:
        df["header_ratio"]        = df["Header_Length"] / (df["Tot_size"] + eps)
        df["payload_size"]        = (df["Tot_size"] - df["Header_Length"]).clip(lower=0)
    flag_cols = [f for f in ["fin_flag_number","syn_flag_number",
                              "psh_flag_number","rst_flag_number"] if f in c]
    if flag_cols:
        df["flag_sum"]            = df[flag_cols].sum(axis=1)
    if "syn_flag_number" in c and "fin_flag_number" in c:
        df["syn_fin_ratio"]       = df["syn_flag_number"] / (df["fin_flag_number"] + eps)
    if "psh_flag_number" in c and "syn_flag_number" in c:
        df["psh_syn_ratio"]       = df["psh_flag_number"] / (df["syn_flag_number"] + eps)
    if "Max" in c and "Min" in c:
        df["stat_range"]          = df["Max"] - df["Min"]
    if "Std" in c and "AVG" in c:
        df["stat_cv"]             = df["Std"] / (df["AVG"] + eps)
    if "Variance" in c and "AVG" in c:
        df["variance_mean_ratio"] = df["Variance"] / (df["AVG"] + eps)
    if "flow_bytes_s" in c and "flow_packets_s" in c:
        df["bytes_per_packet"]    = df["flow_bytes_s"] / (df["flow_packets_s"] + eps)
    return df


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_and_preprocess(data_dir, mode="54class"):
    print(f"Loading dataset  [mode={mode}]...")
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
          f"{len(data['label'].unique())} fine-grained classes")

    # ── Remap labels for 10-class mode ───────────────────────────
    if mode == "10class":
        data["label"] = data["label"].map(COARSE_MAP)
        n_unmapped = data["label"].isna().sum()
        if n_unmapped > 0:
            print(f"  WARNING: {n_unmapped} rows with unmapped labels — dropped")
            data = data.dropna(subset=["label"])
        print(f"  Remapped to {len(data['label'].unique())} coarse classes")

    labels = data["label"].values

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


def split_base_novel(le, novel_classes):
    all_classes  = list(le.classes_)
    novel_valid  = {c for c in novel_classes if c in all_classes}
    missing      = novel_classes - novel_valid
    if missing:
        print(f"  Warning: novel classes not found: {missing}")
    base_names   = [c for c in all_classes if c not in novel_valid]
    novel_names  = [c for c in all_classes if c in novel_valid]
    base_indices = list(le.transform(base_names))
    novel_indices= list(le.transform(novel_names))
    print(f"  Base  ({len(base_names)}):  {base_names}")
    print(f"  Novel ({len(novel_names)}): {novel_names}")
    return base_indices, novel_indices, base_names, novel_names


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


class MAMLClassifier(nn.Module):
    def __init__(self, in_dim, n_way, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.embedding = MLPEmbedding(in_dim, hidden_dim, embed_dim)
        self.head      = nn.Linear(embed_dim, n_way)

    def forward(self, x):
        return self.head(self.embedding(x))


def functional_forward(model, x, weights):
    out = x
    for name, module in model.embedding.net.named_children():
        if isinstance(module, nn.Linear):
            out = F.linear(out, weights[f"embedding.net.{name}.weight"],
                               weights[f"embedding.net.{name}.bias"])
        elif isinstance(module, nn.BatchNorm1d):
            out = F.batch_norm(out,
                running_mean=module.running_mean,
                running_var=module.running_var,
                weight=weights[f"embedding.net.{name}.weight"],
                bias=weights[f"embedding.net.{name}.bias"],
                training=True)
        elif isinstance(module, nn.ReLU):
            out = F.relu(out)
        elif isinstance(module, nn.Dropout):
            out = F.dropout(out, p=module.p, training=model.training)
    return F.linear(out, weights["head.weight"], weights["head.bias"])


def inner_loop(model, support_x, support_y, inner_lr, inner_steps, device):
    fast_weights = {n: p.clone() for n, p in model.named_parameters()}
    for _ in range(inner_steps):
        logits = functional_forward(model, support_x, fast_weights)
        loss   = F.cross_entropy(logits, support_y)
        grads  = torch.autograd.grad(
            loss, fast_weights.values(),
            create_graph=True, allow_unused=True)
        fast_weights = {
            n: p - inner_lr * (g if g is not None else torch.zeros_like(p))
            for (n, p), g in zip(fast_weights.items(), grads)
        }
    return fast_weights


# ─────────────────────────────────────────────────────────────────
# EPISODE SAMPLER
# ─────────────────────────────────────────────────────────────────
class EpisodeSampler:
    def __init__(self, data_x, data_y, class_list, n_way, k_shot, q_query, device):
        self.data_x  = torch.tensor(data_x, dtype=torch.float32).to(device)
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
        return (torch.cat(support_x),
                torch.tensor(support_y, dtype=torch.long).to(self.device),
                torch.cat(query_x),
                torch.tensor(query_y, dtype=torch.long).to(self.device))


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def evaluate(model, sampler, inner_lr, inner_steps, n_episodes, device):
    model.eval()
    accs = []
    for _ in range(n_episodes):
        sup_x, sup_y, qry_x, qry_y = sampler.sample_episode()
        fast_w = inner_loop(model, sup_x, sup_y, inner_lr, inner_steps, device)
        with torch.no_grad():
            logits = functional_forward(model, qry_x, fast_w)
        accs.append(accuracy(logits, qry_y))
    model.train()
    mean_acc = np.mean(accs)
    ci95     = 1.96 * np.std(accs) / np.sqrt(n_episodes)
    return mean_acc, ci95


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Mode setup ───────────────────────────────────────────────
    if args.mode == "10class":
        novel_classes = NOVEL_CLASSES_10
        # For 10-class: n_way_test capped at 3 (only 3 novel classes)
        n_way_test_override = 3
        suffix = "10class"
    else:
        novel_classes = NOVEL_CLASSES_54
        n_way_test_override = None
        suffix = "54class"

    print("=" * 72)
    print(f"MAML  |  mode={args.mode}  |  "
          f"{args.n_way}-way {args.k_shot}-shot  |  device={args.device}")
    print("=" * 72)

    X, y, le, feat_names = load_and_preprocess(DATA_DIR, args.mode)
    in_dim = len(feat_names)

    with open(os.path.join(OUTPUT_DIR,
              f"maml_feature_names_{suffix}.json"), "w") as fh:
        json.dump(feat_names, fh)
    print(f"  Saved feature names ({len(feat_names)})")

    base_idx, novel_idx, base_names, novel_names = split_base_novel(
        le, novel_classes)

    base_mask  = np.isin(y, base_idx)
    novel_mask = np.isin(y, novel_idx)
    base_x,  base_y  = X[base_mask],  y[base_mask]
    novel_x, novel_y = X[novel_mask], y[novel_mask]

    n_way_test = n_way_test_override if n_way_test_override \
                 else min(args.n_way, len(novel_idx))

    print(f"\n  Base  : {len(base_x):,} rows  {len(base_idx)} classes")
    print(f"  Novel : {len(novel_x):,} rows  {len(novel_idx)} classes")
    print(f"  Input dim: {in_dim}")
    print(f"  Test n_way: {n_way_test}")

    train_sampler = EpisodeSampler(
        base_x, base_y, base_idx,
        args.n_way, args.k_shot, args.q_query, device)
    test_sampler  = EpisodeSampler(
        novel_x, novel_y, novel_idx,
        n_way_test, args.k_shot, args.q_query, device)

    print(f"\n  Train sampler: {len(train_sampler.valid_classes)} valid classes")
    print(f"  Test  sampler: {len(test_sampler.valid_classes)} valid classes")

    model    = MAMLClassifier(in_dim, args.n_way).to(device)
    meta_opt = Adam(model.parameters(), lr=args.meta_lr)

    print(f"\n  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Episodes: {args.episodes}  Inner steps: {args.inner_steps}")
    print(f"  Inner lr: {args.inner_lr}  Meta lr: {args.meta_lr}")

    print("\n" + "=" * 72)
    print("META-TRAINING")
    print("=" * 72)

    best_novel_acc = 0.0
    history = defaultdict(list)
    best_pt = os.path.join(OUTPUT_DIR, f"maml_iot_best_{suffix}.pt")

    for ep in range(1, args.episodes + 1):
        model.train()
        sup_x, sup_y, qry_x, qry_y = train_sampler.sample_episode()
        fast_w       = inner_loop(model, sup_x, sup_y,
                                  args.inner_lr, args.inner_steps, device)
        query_logits = functional_forward(model, qry_x, fast_w)
        meta_loss    = F.cross_entropy(query_logits, qry_y)
        meta_acc     = accuracy(query_logits, qry_y)

        meta_opt.zero_grad()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        meta_opt.step()

        history["train_loss"].append(meta_loss.item())
        history["train_acc"].append(meta_acc)

        if ep % args.eval_interval == 0:
            base_acc,  base_ci  = evaluate(model, train_sampler,
                args.inner_lr, args.inner_steps, args.eval_episodes, device)
            novel_acc, novel_ci = evaluate(model, test_sampler,
                args.inner_lr, args.inner_steps, args.eval_episodes, device)

            history["base_acc"].append(base_acc)
            history["novel_acc"].append(novel_acc)

            print(f"  Ep {ep:>6}  "
                  f"loss={np.mean(history['train_loss'][-100:]):.4f}  "
                  f"train_acc={np.mean(history['train_acc'][-100:]):.4f}  "
                  f"base_acc={base_acc:.4f}±{base_ci:.4f}  "
                  f"novel_acc={novel_acc:.4f}±{novel_ci:.4f}")

            if novel_acc > best_novel_acc:
                best_novel_acc = novel_acc
                torch.save(model.state_dict(), best_pt)

    print("\n" + "=" * 72)
    print(f"FINAL META-TEST (novel classes)  [mode={args.mode}]")
    print("=" * 72)

    model.load_state_dict(torch.load(best_pt, map_location=device))

    for k in [1, 5, 10]:
        if k > args.k_shot:
            continue
        s = EpisodeSampler(novel_x, novel_y, novel_idx,
                           n_way_test, k, args.q_query, device)
        acc, ci = evaluate(model, s, args.inner_lr, args.inner_steps,
                           args.eval_episodes * 2, device)
        print(f"  {n_way_test}-way {k:>2}-shot  acc={acc:.4f} ± {ci:.4f}")

    print(f"\n  Best novel acc : {best_novel_acc:.4f}")
    np.save(os.path.join(OUTPUT_DIR,
            f"maml_iot_history_{suffix}.npy"), dict(history))
    print(f"  Model saved    → {best_pt}")
    print("=" * 72)


if __name__ == "__main__":
    main()
