"""
benchmark_models.py — adapted for custom IoT dataset
=====================================================
Trains and evaluates 6 models on your balanced dataset:
  - XGBoost
  - Random Forest
  - Decision Tree
  - MLP  (sklearn)
  - CNN  (PyTorch 1-D conv, requires torch)
  - LSTM (PyTorch, requires torch)

Two modes:
  --mode finegrained  : all specific labels (50 classes)
  --mode coarse       : major categories only (9 classes)
  --mode both         : run both (default)

Usage:
  python benchmark_models.py
  python benchmark_models.py --data G:/iot_dataset/balanced_dataset
  python benchmark_models.py --mode finegrained
"""

import os, glob, argparse, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

# PyTorch optional
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[torch] available — device={DEVICE}')
except ImportError:
    TORCH_OK = False
    print('[torch] NOT found — CNN and LSTM will be skipped.')

# =============================================================================
# CONFIG
# =============================================================================
TEST_SIZE    = 0.25
RANDOM_STATE = 42
N_JOBS       = -1
OUT_DIR      = './benchmark_results'
TOP_FEATS    = 20

# PyTorch training
PT_EPOCHS   = 60
PT_BATCH    = 512
PT_LR       = 5e-4
PT_PATIENCE = 8

# =============================================================================
# COARSE LABEL MAPPING  (fine-grained → major category)
# =============================================================================
COARSE_MAP = {
    # Benign
    'Benign':                       'Benign',
    'Benign_SSH':                   'Benign',
    'Benign_Grafana':               'Benign',
    # DoS
    'DoS_SYN_Flood':                'DoS',
    'DoS_TCP_Flood':                'DoS',
    'DoS_UDP_Flood':                'DoS',
    'DoS_ICMP_Flood':               'DoS',
    'DoS_PSHACK_Flood':             'DoS',
    'DoS_RSTFIN_Flood':             'DoS',
    # DDoS
    'DDoS_SYN_Flood':               'DDoS',
    'DDoS_TCP_Flood':               'DDoS',
    'DDoS_UDP_Flood':               'DDoS',
    'DDoS_ICMP_Flood':              'DDoS',
    'DDoS_PSHACK_Flood':            'DDoS',
    'DDoS_RSTFIN_Flood':            'DDoS',
    'DDoS_ACK_Fragmentation':       'DDoS',
    'DDoS_UDP_Fragmentation':       'DDoS',
    # Mirai
    'Mirai_UDPPlain':               'Mirai',
    'Mirai_GREIP_Flood':            'Mirai',
    'Mirai_Greeth_Flood':           'Mirai',
    'Backdoor_C2_Mirai':            'Mirai',
    # BruteForce
    'BruteForce_Telnet':            'BruteForce',
    'BruteForce_SSH':               'BruteForce',
    'BruteForce_FTP':               'BruteForce',
    'BruteForce_HTTP':              'BruteForce',
    # Recon
    'Recon_Nmap_SYN':               'Recon',
    'Recon_Nmap_Service':           'Recon',
    'Recon_Nmap_OS':                'Recon',
    'Recon_VulScan':                'Recon',
    'Recon_Ping':                   'Recon',
    'Recon_Masscan':                'Recon',
    'Recon_Web':                    'Recon',
    # Spoofing
    'Spoofing_SSH':                 'Spoofing',
    'Spoofing_FTP':                 'Spoofing',
    'Spoofing_Telnet':              'Spoofing',
    'Spoofing_HTTP_Alt':            'Spoofing',
    'Spoofing_Flask':               'Spoofing',
    'Spoofing_HTTP':                'Spoofing',
    'Spoofing_Sensor_Scapy':        'Spoofing',
    'Spoofing_MitM_ARP_Scapy':      'Spoofing',
    # Web
    'Web_SQLi':                     'Web_Based',
    'Web_CmdInjection':             'Web_Based',
    'Web_XSS':                      'Web_Based',
    'Web_Traversal':                'Web_Based',
    'Web_FileUpload':               'Web_Based',
    'Web_ConfigTamper':             'Web_Based',
    'Web_Firmware':                 'Web_Based',
    'Web_UserEnum':                 'Web_Based',
    # False Data Injection
    'FalseDataInjection_Sensor':    'False_Data_Injection',
    'FalseDataInjection_Replay':    'False_Data_Injection',
    'Replay_Sensor_Scapy':          'False_Data_Injection',
    # Backdoor
    'Backdoor_ReverseShell':        'Backdoor',
    'Backdoor_Exfiltration':        'Backdoor',
    'Backdoor_C2Beacon':            'Backdoor',
}

# Columns to drop (metadata — not ML features)
DROP_COLS = {
    'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
    'service', 'label', 'Label'
}

# =============================================================================
# PLOT STYLE
# =============================================================================
PALETTE = {
    'XGBoost':      '#1565C0',
    'RandomForest': '#1E88E5',
    'DecisionTree': '#42A5F5',
    'MLP':          '#0D47A1',
    'CNN':          '#29B6F6',
    'LSTM':         '#4FC3F7',
}
BG    = '#FFFFFF'
FG    = '#0D1B2A'
GRIDC = '#BBDEFB'
SPINE = '#90CAF9'
AXBG  = '#E3F2FD'

plt.rcParams.update({
    'figure.facecolor': BG,    'axes.facecolor':  AXBG,
    'axes.edgecolor':   SPINE, 'axes.labelcolor': FG,
    'axes.titlecolor':  FG,    'xtick.color':     FG,
    'ytick.color':      FG,    'text.color':      FG,
    'grid.color':       GRIDC, 'grid.linewidth':  0.7,
    'font.family':      'DejaVu Sans', 'font.size': 10,
    'axes.titlesize':   13,    'axes.titleweight': 'bold',
    'legend.facecolor': BG,    'legend.edgecolor': SPINE,
    'figure.dpi':       120,
})

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# PYTORCH MODELS  (unchanged from original)
# =============================================================================
if TORCH_OK:
    class TabCNN(nn.Module):
        def __init__(self, n_features, n_classes):
            super().__init__()
            self.pad_to = ((n_features + 3) // 4) * 4
            def conv_block(in_ch, out_ch, k):
                return nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(out_ch, track_running_stats=False),
                    nn.GELU(), nn.Dropout(0.1))
            self.stem   = conv_block(1, 64, 7)
            self.block1 = conv_block(64, 128, 5)
            self.skip1  = nn.Conv1d(64, 128, 1)
            self.pool1  = nn.MaxPool1d(2)
            self.block2 = conv_block(128, 256, 3)
            self.skip2  = nn.Conv1d(128, 256, 1)
            self.block3 = conv_block(256, 256, 3)
            self.gap    = nn.AdaptiveAvgPool1d(1)
            self.head   = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, n_classes))

        def forward(self, x):
            pad = self.pad_to - x.shape[1]
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, pad))
            x = x.unsqueeze(1)
            x = self.stem(x)
            x = self.pool1(self.block1(x) + self.skip1(x))
            x = self.block2(x) + self.skip2(x)
            x = self.block3(x) + x
            x = self.gap(x)
            return self.head(x)

    class TabLSTM(nn.Module):
        def __init__(self, n_features, n_classes):
            super().__init__()
            H = 128
            self.embed = nn.Sequential(nn.Linear(1, 32), nn.GELU())
            self.lstm  = nn.LSTM(input_size=32, hidden_size=H,
                                 num_layers=2, batch_first=True,
                                 dropout=0.3, bidirectional=True)
            self.attn  = nn.Linear(H * 2, 1)
            self.head  = nn.Sequential(
                nn.Linear(H * 2, 256), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(256, 128),   nn.GELU(), nn.Dropout(0.3),
                nn.Linear(128, n_classes))

        def forward(self, x):
            x = x.unsqueeze(-1)
            x = self.embed(x)
            out, _ = self.lstm(x)
            w   = torch.softmax(self.attn(out), dim=1)
            ctx = (out * w).sum(dim=1)
            return self.head(ctx)

    class TorchWrapper:
        def __init__(self, model_cls, n_features, n_classes):
            self.model_cls  = model_cls
            self.n_features = n_features
            self.n_classes  = n_classes
            self.model      = None

        def fit(self, X, y):
            self.model = self.model_cls(self.n_features, self.n_classes).to(DEVICE)
            opt = torch.optim.AdamW(self.model.parameters(), lr=PT_LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PT_EPOCHS)
            criterion = nn.CrossEntropyLoss()
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            n_val = max(1, int(0.1 * len(X_t)))
            idx = torch.randperm(len(X_t))
            tr_idx, va_idx = idx[n_val:], idx[:n_val]
            tr_dl = DataLoader(TensorDataset(X_t[tr_idx], y_t[tr_idx]),
                               batch_size=PT_BATCH, shuffle=True)
            va_dl = DataLoader(TensorDataset(X_t[va_idx], y_t[va_idx]),
                               batch_size=PT_BATCH * 2)
            best_val, patience_cnt, best_state = 1e9, 0, None
            for epoch in range(1, PT_EPOCHS + 1):
                self.model.train()
                for xb, yb in tr_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(self.model(xb), yb)
                    loss.backward(); opt.step()
                scheduler.step()
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in va_dl:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        val_loss += criterion(self.model(xb), yb).item()
                val_loss /= len(va_dl)
                if val_loss < best_val - 1e-4:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= PT_PATIENCE:
                        print(f' (early stop ep {epoch})', end='')
                        break
            if best_state:
                self.model.load_state_dict(
                    {k: v.to(DEVICE) for k, v in best_state.items()})
            return self

        def predict(self, X):
            self.model.eval()
            X_t = torch.tensor(X, dtype=torch.float32)
            dl  = DataLoader(TensorDataset(X_t),
                             batch_size=PT_BATCH * 4, shuffle=False)
            preds = []
            with torch.no_grad():
                for (xb,) in dl:
                    preds.append(self.model(xb.to(DEVICE)).argmax(1).cpu())
            return torch.cat(preds).numpy()

        def get_feature_importances(self, X, y, n_samples=2000):
            self.model.eval()
            idx  = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
            X_np = X[idx].astype(np.float32)
            y_t  = torch.tensor(y[idx], dtype=torch.long).to(DEVICE)
            with torch.enable_grad():
                X_t = torch.tensor(X_np, requires_grad=True).to(DEVICE)
                self.model.train()
                logits = self.model(X_t)
                loss   = nn.CrossEntropyLoss()(logits, y_t)
                loss.backward()
                self.model.eval()
            if X_t.grad is not None:
                imps = X_t.grad.abs().mean(0).cpu().numpy()
            else:
                print(' (grad=None, permutation fallback)', end='')
                self.model.eval()
                n_feat = X_np.shape[1]
                imps   = np.zeros(n_feat)
                with torch.no_grad():
                    X_base    = torch.tensor(X_np).to(DEVICE)
                    base_pred = self.model(X_base).argmax(1)
                    for fi in range(n_feat):
                        X_perm        = X_base.clone()
                        idx_perm      = torch.randperm(len(X_perm))
                        X_perm[:, fi] = X_perm[idx_perm, fi]
                        perm_pred     = self.model(X_perm).argmax(1)
                        imps[fi]      = (perm_pred != base_pred).float().mean().item()
            return imps / (imps.max() + 1e-10)


# =============================================================================
# DATA LOADING  — adapted for custom dataset
# =============================================================================
def load_data(data_path, mode):
    """
    Load all CSVs recursively from subfolders.
    mode='finegrained' : use label column as-is (50 classes)
    mode='coarse'      : map to major categories via COARSE_MAP (9 classes)
    """
    files = sorted(glob.glob(os.path.join(data_path, '**', '*.csv'), recursive=True))
    if not files:
        raise FileNotFoundError(f'No CSV files found in {data_path}')

    print(f'\n[DATA] Loading {len(files)} files — mode={mode}')
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'label' not in df.columns:
                print(f'  SKIP (no label): {os.path.relpath(f, data_path)}')
                continue
            dfs.append(df)
            print(f'  {len(df):>8,}  {os.path.relpath(f, data_path)}')
        except Exception as e:
            print(f'  ERROR {os.path.basename(f)}: {e}')

    data = pd.concat(dfs, ignore_index=True)
    print(f'\nTotal loaded: {len(data):,} rows')

    # Apply label mapping
    if mode == 'coarse':
        data['label'] = data['label'].map(COARSE_MAP)
        unmapped = data['label'].isna().sum()
        if unmapped > 0:
            print(f'  Warning: {unmapped:,} rows with unmapped labels dropped')
        data = data.dropna(subset=['label'])
    # finegrained: keep labels as-is

    # Drop metadata + non-numeric columns
    feat_cols = [c for c in data.columns if c not in DROP_COLS]
    X = (data[feat_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0)
         .select_dtypes(include=[np.number]))

    # Remove constant features
    std = X.std()
    const = std[std == 0].index.tolist()
    if const:
        print(f'  Dropping {len(const)} constant features: {const}')
        X = X.drop(columns=const)

    le = LabelEncoder()
    y  = le.fit_transform(data['label'])

    print(f'\nClass distribution:')
    print(data['label'].value_counts().to_string())
    print(f'\nFeatures: {X.shape[1]} | Classes: {len(le.classes_)}')
    print(f'Classes: {list(le.classes_)}')

    return X.values, y, le, list(X.columns)


# =============================================================================
# MODEL REGISTRY
# =============================================================================
def get_models(n_classes, n_features):
    models = {
        'XGBoost': (False, XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
            eval_metric='mlogloss', random_state=RANDOM_STATE,
            n_jobs=N_JOBS, tree_method='hist')),
    }
    return models


# =============================================================================
# TRAIN & EVALUATE
# =============================================================================
def evaluate_model(name, needs_scale, model, X_tr, X_te, y_tr, y_te,
                   X_tr_sc, X_te_sc, le):
    Xtr = X_tr_sc if needs_scale else X_tr
    Xte = X_te_sc if needs_scale else X_te
    print(f'\n  [{name}] training...', end='', flush=True)
    t0 = time.time()
    model.fit(Xtr, y_tr)
    t_train = time.time() - t0
    y_pred = model.predict(Xte)
    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
    print(f'  {t_train:.1f}s | Acc={acc:.4f}  F1={f1:.4f}')
    return {'name': name, 'model': model, 'needs_scale': needs_scale,
            'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec,
            'y_pred': y_pred, 'train_time': t_train,
            'report': classification_report(y_te, y_pred,
                        target_names=le.classes_, zero_division=0)}


# =============================================================================
# PLOTS
# =============================================================================
def plot_comparison(results, mode):
    tag   = '54-Class' if mode == 'finegrained' else '10-Class'
    names = [r['name'] for r in results]
    metrics = [('acc', 'Accuracy'), ('f1', 'F1 (macro)'),
               ('prec', 'Precision'), ('rec', 'Recall')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.patch.set_facecolor(BG)
    fig.suptitle(f'Model Comparison — {tag} IoT Intrusion Detection',
                 fontsize=15, fontweight='bold', color=FG, y=1.01)

    for ax, (key, label) in zip(axes, metrics):
        vals  = [r[key] for r in results]
        order = np.argsort(vals)[::-1]
        ranks = np.empty(len(vals), dtype=int)
        ranks[order] = np.arange(len(vals))
        cmap  = plt.cm.Blues
        cvals = [cmap(0.95 - 0.60 * (ranks[i] / max(len(vals)-1, 1)))
                 for i in range(len(vals))]
        bars = ax.bar(names, vals, color=cvals,
                      edgecolor='white', linewidth=0.8, width=0.55)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9, color=FG,
                    fontweight='bold' if ranks[i] == 0 else 'normal')
        ax.set_title(label, pad=10, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
        ax.yaxis.grid(True, alpha=0.5, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(SPINE)

    plt.tight_layout(pad=2.5)
    path = os.path.join(OUT_DIR, f'comparison_{mode}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved: {path}')


def plot_confusion(result, y_te, le, mode):
    cm      = confusion_matrix(y_te, result['y_pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n       = len(le.classes_)
    tag     = '54-Class' if mode == 'finegrained' else '10-Class'

    cmap  = LinearSegmentedColormap.from_list(
        'blue_heat', ['#FFFFFF', '#BBDEFB', '#1E88E5', '#1565C0', '#0D47A1'])
    fig_w = max(10, n * 0.75)
    fig_h = max(8,  n * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, linewidths=0.5, linecolor='#DDDDDD',
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 7 if n > 15 else 9, 'color': FG})
    ax.set_xlabel('Predicted', labelpad=12, fontsize=11, color=FG)
    ax.set_ylabel('True',      labelpad=12, fontsize=11, color=FG)
    ax.set_title(f'Confusion Matrix — {result["name"]} ({tag})\n'
                 f'Acc={result["acc"]:.4f}  F1={result["f1"]:.4f}',
                 pad=14, fontsize=13, color=FG)
    ax.tick_params(axis='x', rotation=45,
                   labelsize=7 if n > 15 else 9, colors=FG)
    ax.tick_params(axis='y', rotation=0,
                   labelsize=7 if n > 15 else 9, colors=FG)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'confusion_{mode}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved: {path}')


def get_importances(result, X_tr, y_tr, X_tr_sc, feat_names):
    model = result['model']
    name  = result['name']
    if hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
    elif isinstance(model, MLPClassifier):
        imps = np.abs(model.coefs_[0]).mean(axis=1)
    elif TORCH_OK and isinstance(model, TorchWrapper):
        X_use = X_tr_sc if result['needs_scale'] else X_tr
        imps  = model.get_feature_importances(X_use, y_tr)
    else:
        return None
    imps = imps / (imps.max() + 1e-10)
    idx  = np.argsort(imps)[::-1][:TOP_FEATS]
    return [(feat_names[i], imps[i]) for i in idx]


def plot_feature_importance(result, X_tr, y_tr, X_tr_sc, feat_names, mode):
    tag  = '54-Class' if mode == 'finegrained' else '10-Class'
    name = result['name']
    data = get_importances(result, X_tr, y_tr, X_tr_sc, feat_names)
    if data is None:
        print(f'  [{name}] no importances — skipped')
        return
    feats, imps = zip(*data)
    n = len(feats)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.40)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AXBG)
    colors = plt.cm.Blues(np.linspace(0.3, 0.95, n))
    bars = ax.barh(range(n), imps[::-1], color=colors,
                   edgecolor='white', linewidth=0.5, height=0.68)
    ax.set_yticks(range(n))
    ax.set_yticklabels(feats[::-1], fontsize=9, color=FG)
    ax.set_xlabel('Normalised Importance', labelpad=10, color=FG)
    ax.set_title(f'Top {n} Feature Importances — {name} ({tag})',
                 pad=12, fontsize=13, color=FG)
    ax.xaxis.grid(True, alpha=0.5, linestyle='--', color=GRIDC)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(SPINE)
    for bar, v in zip(bars, imps[::-1]):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=8, color=FG)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'feature_importance_{mode}_{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved: {path}')


def print_summary(results, mode):
    tag = '54-Class' if mode == 'finegrained' else '10-Class'
    print(f'\n{"="*68}')
    print(f'  FINAL — {tag}')
    print(f'  {"Model":<16} {"Acc":>8} {"F1":>8} {"Prec":>8} '
          f'{"Rec":>8} {"Time":>8}')
    print(f'  {"-"*60}')
    for r in sorted(results, key=lambda x: -x['f1']):
        print(f'  {r["name"]:<16} {r["acc"]:>8.4f} {r["f1"]:>8.4f} '
              f'{r["prec"]:>8.4f} {r["rec"]:>8.4f} '
              f'{r["train_time"]:>7.1f}s')
    best = max(results, key=lambda x: x['f1'])
    print(f'\n  Best model: {best["name"]}  (F1={best["f1"]:.4f})')
    print(f'{"="*68}')


# =============================================================================
# MAIN
# =============================================================================
def run(data_path, mode):
    print(f'\n{"="*68}')
    print(f'  BENCHMARK — {mode.upper()}')
    print(f'{"="*68}')

    X, y, le, feat_names = load_data(data_path, mode)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    models  = get_models(len(le.classes_), X.shape[1])
    results = []
    for name, (needs_scale, model) in models.items():
        res = evaluate_model(name, needs_scale, model,
                             X_tr, X_te, y_tr, y_te,
                             X_tr_sc, X_te_sc, le)
        results.append(res)

    print_summary(results, mode)

    print('\n[PLOTS]')
    plot_comparison(results, mode)
    best = max(results, key=lambda x: x['f1'])
    print(f'  Confusion matrix for best model: {best["name"]}')
    plot_confusion(best, y_te, le, mode)

    for res in results:
        plot_feature_importance(res, X_tr, y_tr, X_tr_sc, feat_names, mode)

    rpt = os.path.join(OUT_DIR, f'report_{mode}.txt')
    with open(rpt, 'w') as f:
        for r in sorted(results, key=lambda x: -x['f1']):
            f.write(f'\n{"="*60}\n{r["name"]}\n{"="*60}\n{r["report"]}')
    print(f'  Saved: {rpt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=r'G:\iot_dataset\balanced_dataset')
    parser.add_argument('--mode', choices=['finegrained', 'coarse', 'both'],
                        default='both')
    args = parser.parse_args()

    if args.mode in ('finegrained', 'both'):
        run(args.data, 'finegrained')
    if args.mode in ('coarse', 'both'):
        run(args.data, 'coarse')

    print(f'\n[DONE] Results in: {os.path.abspath(OUT_DIR)}')