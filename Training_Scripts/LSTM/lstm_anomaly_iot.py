"""
lstm_anomaly_iot_pytorch.py
============================
LSTM Autoencoder for IoT sensor anomaly detection — PyTorch version.

Strategy: train one autoencoder per NORMAL sensor, then evaluate
each against anomaly sensors. This handles sensors in different
physical locations (different temp/humidity ranges) correctly.

Normal  : 192.168.1.115, 127, 161, 173, 180
Anomaly : 192.168.1.156, 170, 181, 231
Skip    : 192.168.1.116, 128, 162

Usage:
  python lstm_anomaly_iot_pytorch.py
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
SENSORS_DIR = r"G:\iot_dataset\database\sensors"
OUTPUT_DIR  = r"G:\iot_dataset\lstm_results"
DATE_START  = "2026-03-01"
DATE_END    = "2026-04-13"

NORMAL_IPS  = {"192.168.1.115", "192.168.1.127", "192.168.1.161",
               "192.168.1.173", "192.168.1.180"}
ANOMALY_IPS = {"192.168.1.156", "192.168.1.170", "192.168.1.181",
               "192.168.1.231"}
SKIP_IPS    = {"192.168.1.116", "192.168.1.128", "192.168.1.162"}

# Hyperparameters
SEQUENCE_LENGTH         = 10
LSTM_UNITS              = 32
ENCODING_DIM            = 16
EPOCHS                  = 150
BATCH_SIZE              = 32
VALIDATION_SPLIT        = 0.2
EARLY_STOPPING_PATIENCE = 20
THRESHOLD_PERCENTILE    = 95
LR                      = 1e-3

TEMP_MIN, TEMP_MAX = -10, 50
HUM_MIN,  HUM_MAX  = 0, 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def ip_from_filename(fname):
    base = os.path.basename(fname).replace("sensor_", "").replace(".csv", "")
    return base.replace("_", ".")


def load_sensor(filepath):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timestamp"] >= DATE_START) & (df["timestamp"] <= DATE_END)]
    # Pivot: temp and humidity are on separate rows with same timestamp
    df = df.groupby("timestamp", as_index=False).agg(
        temperature=("temperature", "first"),
        humidity=("humidity",    "first"),
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["temperature"] = df["temperature"].ffill()
    df["humidity"]    = df["humidity"].ffill()
    df = df.dropna(subset=["temperature", "humidity"])
    df = df[(df["temperature"] >= TEMP_MIN) & (df["temperature"] <= TEMP_MAX) &
            (df["humidity"]    >= HUM_MIN)  & (df["humidity"]    <= HUM_MAX)]
    return df[["timestamp", "temperature", "humidity"]]


def load_all_sensors():
    files = sorted(glob.glob(os.path.join(SENSORS_DIR, "sensor_*.csv")))
    files = [f for f in files if "sensor_all" not in f]
    normal_dfs, anomaly_dfs = [], []
    for f in files:
        ip = ip_from_filename(f)
        if ip in SKIP_IPS:
            print(f"  SKIP    {ip}")
            continue
        df = load_sensor(f)
        if len(df) < SEQUENCE_LENGTH + 10:
            print(f"  SKIP    {ip} — too few rows ({len(df)})")
            continue
        if ip in NORMAL_IPS:
            normal_dfs.append((ip, df))
            print(f"  NORMAL  {ip:<20} {len(df):>6,} rows")
        elif ip in ANOMALY_IPS:
            anomaly_dfs.append((ip, df))
            print(f"  ANOMALY {ip:<20} {len(df):>6,} rows")
    return normal_dfs, anomaly_dfs


# ─────────────────────────────────────────────────────────────────
# SEQUENCES
# ─────────────────────────────────────────────────────────────────
def create_sequences(data, seq_length):
    seqs = [data[i:i+seq_length] for i in range(len(data) - seq_length + 1)]
    return np.array(seqs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, lstm_units, encoding_dim, seq_length):
        super().__init__()
        self.seq_length    = seq_length
        self.enc_lstm1     = nn.LSTM(n_features,  lstm_units,   batch_first=True)
        self.enc_lstm2     = nn.LSTM(lstm_units,  encoding_dim, batch_first=True)
        self.dec_lstm1     = nn.LSTM(encoding_dim, encoding_dim, batch_first=True)
        self.dec_lstm2     = nn.LSTM(encoding_dim, lstm_units,   batch_first=True)
        self.output_layer  = nn.Linear(lstm_units, n_features)
        self.relu          = nn.ReLU()

    def forward(self, x):
        out, _      = self.enc_lstm1(x);           out = self.relu(out)
        _, (h, _)   = self.enc_lstm2(out)
        bottleneck  = self.relu(h.squeeze(0))
        repeated    = bottleneck.unsqueeze(1).repeat(1, self.seq_length, 1)
        out, _      = self.dec_lstm1(repeated);    out = self.relu(out)
        out, _      = self.dec_lstm2(out);         out = self.relu(out)
        return self.output_layer(out)


# ─────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────
def train_model(X_train_seq, X_val_seq, ip):
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val_seq)),
                              batch_size=BATCH_SIZE, shuffle=False)

    model     = LSTMAutoencoder(2, LSTM_UNITS, ENCODING_DIM,
                                SEQUENCE_LENGTH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val  = float("inf")
    patience_cnt = 0
    best_state   = None
    history   = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for xb, in train_loader:
            xb   = xb.to(DEVICE)
            loss = criterion(model(xb), xb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, in val_loader:
                val_losses.append(criterion(model(xb.to(DEVICE)),
                                            xb.to(DEVICE)).item())

        tl = np.mean(tr_losses)
        vl = np.mean(val_losses)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if vl < best_val - 1e-6:
            best_val, patience_cnt = vl, 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stop ep {epoch}  val_loss={vl:.6f}")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, history


# ─────────────────────────────────────────────────────────────────
# RECONSTRUCTION ERRORS
# ─────────────────────────────────────────────────────────────────
def recon_errors(model, X_seq):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_seq)),
                        batch_size=256, shuffle=False)
    errs = []
    with torch.no_grad():
        for xb, in loader:
            xb = xb.to(DEVICE)
            errs.append(((xb - model(xb))**2).mean(dim=(1,2)).cpu().numpy())
    return np.concatenate(errs)


# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────
def plot_all_histories(histories, output_dir):
    n    = len(histories)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows),
                             sharey=False)
    axes = np.array(axes).flatten()
    for ax, (ip, hist) in zip(axes, histories):
        ax.plot(hist["train_loss"], label="Train", color="#2E86AB")
        ax.plot(hist["val_loss"],   label="Val",
                color="#E84855", linestyle="--")
        ax.set_title(f"{ip}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Training History — One Model per Normal Sensor",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: training_history.png")


def plot_error_distribution(all_normal_errors, all_anomaly_errors,
                             threshold, output_dir):
    # Clip x-axis to show meaningful range (exclude extreme outliers)
    p99 = np.percentile(all_anomaly_errors, 99)
    xlim = min(p99 * 1.1, 20.0)  # cap at 20 for readability

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: linear scale clipped
    axes[0].hist(all_normal_errors,  bins=80, alpha=0.7,
                 color="#2E86AB", label="Normal sensors")
    axes[0].hist(all_anomaly_errors, bins=80, alpha=0.7,
                 color="#E84855", label="Anomaly sensors")
    axes[0].axvline(threshold, color="black", linestyle="--",
                    linewidth=2, label=f"Threshold={threshold:.4f}")
    axes[0].set_xlim(0, xlim)
    axes[0].set_xlabel("Reconstruction Error (MSE)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution (clipped x-axis)",
                      fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: log y-scale, full range
    axes[1].hist(all_normal_errors,  bins=80, alpha=0.7,
                 color="#2E86AB", label="Normal sensors")
    axes[1].hist(all_anomaly_errors, bins=80, alpha=0.7,
                 color="#E84855", label="Anomaly sensors")
    axes[1].axvline(threshold, color="black", linestyle="--",
                    linewidth=2, label=f"Threshold={threshold:.4f}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Reconstruction Error (MSE)")
    axes[1].set_ylabel("Count (log scale)")
    axes[1].set_title("Error Distribution (log y-scale)",
                      fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Reconstruction Error Distribution — Normal vs Anomaly",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction_errors.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: reconstruction_errors.png")


def plot_per_sensor_timeline(sensor_models, sensor_scalers,
                              all_dfs, threshold, label, output_dir):
    n = len(all_dfs)
    if n == 0: return
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n))
    if n == 1: axes = [axes]
    color = "#2E86AB" if label == "Normal" else "#E84855"

    for ax, (ip, df) in zip(axes, all_dfs):
        vals = df[["temperature","humidity"]].values.astype(np.float32)
        ts   = df["timestamp"].values[SEQUENCE_LENGTH-1:]

        if ip in sensor_models:
            # Normal sensor — use its own model
            scaler = sensor_scalers[ip]
            scaled = scaler.transform(vals).astype(np.float32)
            seqs   = create_sequences(scaled, SEQUENCE_LENGTH)
            errs   = recon_errors(sensor_models[ip], seqs)
        else:
            # Anomaly sensor — minimum error across all normal models
            all_model_errors = []
            for norm_ip, model in sensor_models.items():
                scaler = sensor_scalers[norm_ip]
                scaled = scaler.transform(vals).astype(np.float32)
                seqs   = create_sequences(scaled, SEQUENCE_LENGTH)
                if len(seqs) == 0:
                    continue
                all_model_errors.append(recon_errors(model, seqs))
            errs = np.min(np.stack(all_model_errors, axis=0), axis=0)

        ax.plot(ts, errs, linewidth=0.8, color=color, alpha=0.85)
        ax.axhline(threshold, color="black", linestyle="--",
                   linewidth=1, label=f"Threshold={threshold:.4f}")
        ax.set_title(f"{ip}  [{label}]", fontsize=10)
        ax.set_ylabel("Recon Error")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(True, alpha=0.2); ax.legend(fontsize=8)

    plt.suptitle(f"Reconstruction Errors Over Time — {label} Sensors",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"per_sensor_{label.lower()}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_metrics(metrics, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    values = [metrics["accuracy"], metrics["precision"],
              metrics["recall"],   metrics["f1"], metrics["auc_roc"]]
    colors = ["#2E86AB","#A23B72","#F18F01","#4CAF50","#9C27B0"]
    bars   = ax.bar(names, values, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.01, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("LSTM Autoencoder — Anomaly Detection Metrics",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: metrics_summary.png")


def plot_per_sensor_boxplot(normal_errors_dict, anomaly_errors_dict,
                             threshold, output_dir):
    """Box plot of reconstruction errors per sensor — shows
    separation between normal and anomaly sensors clearly."""
    all_data   = []
    all_labels = []
    all_colors = []

    for ip, errs in normal_errors_dict.items():
        # Clip to p99 for readability
        clipped = errs[errs <= np.percentile(errs, 99)]
        all_data.append(clipped)
        all_labels.append(f"{ip.split('.')[-1]}\n[N]")
        all_colors.append("#2E86AB")

    for ip, errs in anomaly_errors_dict.items():
        clipped = errs[errs <= np.percentile(errs, 99)]
        all_data.append(clipped)
        all_labels.append(f"{ip.split('.')[-1]}\n[A]")
        all_colors.append("#E84855")

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(all_data, labels=all_labels,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(threshold, color="black", linestyle="--",
               linewidth=1.5, label=f"Threshold $\\tau$={threshold:.4f}")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_xlabel("Sensor IP (last octet)  [N=Normal, A=Anomaly]")
    ax.set_title("Per-Sensor Reconstruction Error Distribution",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_sensor_boxplot.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: per_sensor_boxplot.png")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LSTM Autoencoder — IoT Sensor Anomaly Detection (PyTorch)")
    print("=" * 70)
    print(f"Device:     {DEVICE}")
    print(f"Date range: {DATE_START} → {DATE_END}")
    print()

    normal_dfs, anomaly_dfs = load_all_sensors()
    if not normal_dfs:
        print("ERROR: No normal sensor data."); return
    if not anomaly_dfs:
        print("ERROR: No anomaly sensor data."); return

    # ── Train one model per normal sensor ────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING — one model per normal sensor")
    print("=" * 70)

    sensor_models  = {}
    sensor_scalers = {}
    histories      = []
    val_errors_all = []

    for ip, df in normal_dfs:
        print(f"\n  [{ip}]  {len(df):,} rows")
        vals = df[["temperature","humidity"]].values.astype(np.float32)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(vals)

        n_val       = max(SEQUENCE_LENGTH+1, int(len(X_scaled)*VALIDATION_SPLIT))
        X_train_seq = create_sequences(X_scaled[:-n_val].astype(np.float32),
                                       SEQUENCE_LENGTH)
        X_val_seq   = create_sequences(X_scaled[-n_val:].astype(np.float32),
                                       SEQUENCE_LENGTH)

        print(f"    Train seqs: {len(X_train_seq):,}  "
              f"Val seqs: {len(X_val_seq):,}")

        model, hist = train_model(X_train_seq, X_val_seq, ip)
        sensor_models[ip]  = model
        sensor_scalers[ip] = scaler
        histories.append((ip, hist))

        val_err = recon_errors(model, X_val_seq)
        val_errors_all.append(val_err)
        print(f"    Val mean_err: {val_err.mean():.6f}")

    # Global threshold from all normal val errors
    all_val_errors = np.concatenate(val_errors_all)
    threshold      = float(np.percentile(all_val_errors, THRESHOLD_PERCENTILE))
    print(f"\nGlobal threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")

    # ── Compute errors for all sensors ───────────────────────────
    print("\n" + "=" * 70)
    print("RECONSTRUCTION ERRORS")
    print("=" * 70)

    all_normal_errors  = []
    all_anomaly_errors = []

    # Normal sensors — use their own model
    for ip, df in normal_dfs:
        model  = sensor_models[ip]
        scaler = sensor_scalers[ip]
        vals   = scaler.transform(
            df[["temperature","humidity"]].values).astype(np.float32)
        seqs   = create_sequences(vals, SEQUENCE_LENGTH)
        errs   = recon_errors(model, seqs)
        all_normal_errors.append(errs)
        print(f"  NORMAL  {ip:<20} mean_err={errs.mean():.6f}")

    # Anomaly sensors — evaluate against ALL normal models,
    # take minimum reconstruction error per sequence (Option A).
    # Rationale: a sequence is anomalous if NO normal model
    # can reconstruct it well.
    for ip, df in anomaly_dfs:
        vals  = df[["temperature","humidity"]].values.astype(np.float32)
        # Collect errors from every normal model using its own scaler
        all_model_errors = []
        for norm_ip, model in sensor_models.items():
            scaler = sensor_scalers[norm_ip]
            scaled = scaler.transform(vals).astype(np.float32)
            seqs   = create_sequences(scaled, SEQUENCE_LENGTH)
            if len(seqs) == 0:
                continue
            errs = recon_errors(model, seqs)
            all_model_errors.append(errs)
        # Per-sequence minimum across all models
        min_errors = np.min(np.stack(all_model_errors, axis=0), axis=0)
        all_anomaly_errors.append(min_errors)
        print(f"  ANOMALY {ip:<20} mean_min_err={min_errors.mean():.6f}"
              f"  (min over {len(all_model_errors)} models)")

    normal_errors  = np.concatenate(all_normal_errors)
    anomaly_errors = np.concatenate(all_anomaly_errors)

    # Per-sensor error dicts for box plot
    normal_errors_dict  = {}
    anomaly_errors_dict = {}

    for ip, df in normal_dfs:
        model  = sensor_models[ip]
        scaler = sensor_scalers[ip]
        vals   = scaler.transform(
            df[["temperature","humidity"]].values).astype(np.float32)
        seqs   = create_sequences(vals, SEQUENCE_LENGTH)
        normal_errors_dict[ip] = recon_errors(model, seqs)

    for ip, df in anomaly_dfs:
        vals = df[["temperature","humidity"]].values.astype(np.float32)
        all_model_errors = []
        for norm_ip, model in sensor_models.items():
            scaler = sensor_scalers[norm_ip]
            scaled = scaler.transform(vals).astype(np.float32)
            seqs   = create_sequences(scaled, SEQUENCE_LENGTH)
            if len(seqs) > 0:
                all_model_errors.append(recon_errors(model, seqs))
        anomaly_errors_dict[ip] = np.min(
            np.stack(all_model_errors, axis=0), axis=0)

    # ── Evaluate ──────────────────────────────────────────────────
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    all_labels = np.concatenate([np.zeros(len(normal_errors)),
                                 np.ones(len(anomaly_errors))])
    preds      = (all_errors > threshold).astype(int)

    metrics = {
        "accuracy":          float(accuracy_score(all_labels, preds)),
        "precision":         float(precision_score(all_labels, preds, zero_division=0)),
        "recall":            float(recall_score(all_labels, preds, zero_division=0)),
        "f1":                float(f1_score(all_labels, preds, zero_division=0)),
        "auc_roc":           float(roc_auc_score(all_labels, all_errors)),
        "threshold":         threshold,
        "mean_err_normal":   float(normal_errors.mean()),
        "mean_err_anomaly":  float(anomaly_errors.mean()),
    }

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:<22} {v:.6f}")

    # ── Plots ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PLOTS")
    print("=" * 70)
    plot_all_histories(histories, OUTPUT_DIR)
    plot_error_distribution(normal_errors, anomaly_errors,
                             threshold, OUTPUT_DIR)
    plot_per_sensor_boxplot(normal_errors_dict, anomaly_errors_dict,
                             threshold, OUTPUT_DIR)
    plot_per_sensor_timeline(sensor_models, sensor_scalers,
                              normal_dfs,  threshold, "Normal",  OUTPUT_DIR)
    plot_per_sensor_timeline(sensor_models, sensor_scalers,
                              anomaly_dfs, threshold, "Anomaly", OUTPUT_DIR)
    plot_metrics(metrics, OUTPUT_DIR)

    # Save
    pd.DataFrame([{"Metric": k, "Value": v}
                  for k, v in metrics.items()]).to_csv(
        os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    print("  Saved: metrics.csv")

    for ip, model in sensor_models.items():
        safe = ip.replace(".", "_")
        torch.save(model.state_dict(),
                   os.path.join(OUTPUT_DIR, f"lstm_{safe}.pt"))
    print(f"  Saved {len(sensor_models)} model files")

    print("\n" + "=" * 70)
    print(f"DONE — output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()