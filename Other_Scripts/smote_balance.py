#!/usr/bin/env python3
"""
smote_balance.py
Full preprocessing + SMOTE balancing for IoT dataset.

Strategy:
  Benign: undersample to 1:9 ratio vs total attack samples
  Attack > 28k  → undersample to 30k
  Attack 18k-28k → oversample to 20k
  Attack 10k-18k → oversample to 10k
  Attack < 10k  → oversample to 10k

Steps:
  1. Load all CSVs from subfolders
  2. Remove duplicates
  3. Remove all-zero feature rows
  4. Drop metadata + constant features
  5. SMOTE oversample + random undersample
  6. Save to balanced_dataset/ with same subfolder structure

Usage: python smote_balance.py
"""

import pandas as pd
import numpy as np
import glob
import os
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"G:\iot_dataset\Final_dataset"
OUTPUT_DIR  = r"G:\iot_dataset\balanced_dataset"

# Columns to always drop (metadata)
DROP_META = [
    'timestamp', 'src_ip', 'src_port',
    'dst_ip', 'service'
]

SMOTE_K    = 5
RANDOM_SEED = 42

# Subfolder mapping — label prefix → output subfolder
SUBFOLDER_MAP = {
    'Benign':               'Benign',
    'DoS':                  'DoS',
    'DDoS':                 'DDoS',
    'Mirai':                'Mirai',
    'Recon':                'Recon',
    'BruteForce':           'BruteForce',
    'Web':                  'Web-Based',
    'Spoofing':             'Spoofing',
    'FalseDataInjection':   'False_Data_Injection',
    'Replay':               'False_Data_Injection',
    'Backdoor':             'Backdoor',
}

def get_subfolder(label):
    for prefix, folder in SUBFOLDER_MAP.items():
        if label.startswith(prefix):
            return folder
    return 'Other'


# ── Step 1: Load ──────────────────────────────────────────────────────────────
def load_dataset():
    print("=" * 60)
    print("Step 1: Loading dataset")
    print("=" * 60)
    dfs = []
    files = sorted(glob.glob(os.path.join(DATASET_DIR, '**', '*.csv'), recursive=True))
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'label' not in df.columns:
                print(f"  SKIP (no label): {os.path.relpath(f, DATASET_DIR)}")
                continue
            dfs.append(df)
            print(f"  {len(df):>8,}  {os.path.relpath(f, DATASET_DIR)}")
        except Exception as e:
            print(f"  ERROR {os.path.basename(f)}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined):,} rows, {combined.shape[1]} columns")
    return combined


# ── Step 2: Drop metadata ─────────────────────────────────────────────────────
def drop_metadata(df):
    print("\n" + "=" * 60)
    print("Step 2: Dropping metadata columns")
    print("=" * 60)
    drop = [c for c in DROP_META if c in df.columns]
    df = df.drop(columns=drop)
    print(f"  Dropped: {drop}")
    print(f"  Remaining: {df.shape[1]} columns")
    return df


# ── Step 3: Remove duplicates ─────────────────────────────────────────────────
def remove_duplicates(df):
    print("\n" + "=" * 60)
    print("Step 3: Removing duplicate rows")
    print("=" * 60)
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed: {before - len(df):,} duplicates")
    print(f"  Remaining: {len(df):,} rows")
    return df


# ── Step 4: Remove all-zero rows ──────────────────────────────────────────────
def remove_zero_rows(df):
    print("\n" + "=" * 60)
    print("Step 4: Removing all-zero feature rows")
    print("=" * 60)
    before = len(df)
    feature_cols = [c for c in df.columns if c != 'label']
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    mask = (df[numeric_cols] == 0).all(axis=1)
    df = df[~mask]
    print(f"  Removed: {before - len(df):,} all-zero rows")
    print(f"  Remaining: {len(df):,} rows")
    return df


# ── Step 5: Remove constant features ─────────────────────────────────────────
def remove_constant_features(df):
    print("\n" + "=" * 60)
    print("Step 5: Removing constant (zero-variance) features")
    print("=" * 60)
    label_col = df['label']
    X = df.drop(columns=['label'])

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)

    std = X.std()
    constant = std[std == 0].index.tolist()
    X = X.drop(columns=constant)
    print(f"  Removed {len(constant)} constant features: {constant}")
    print(f"  Remaining features: {X.shape[1]}")

    df_out = X.copy()
    df_out['label'] = label_col.values
    return df_out, constant


# ── Step 6: Balance ───────────────────────────────────────────────────────────
def compute_targets(counts):
    """Compute target sample count per class."""
    attack_labels = {l: c for l, c in counts.items() if not l.startswith('Benign')}
    total_attack  = sum(attack_labels.values())

    # Benign target = total_attack / 9
    benign_target = max(1000, total_attack // 9)

    targets = {}
    for label, count in counts.items():
        if label.startswith('Benign'):
            targets[label] = benign_target // max(1, sum(
                1 for l in counts if l.startswith('Benign')))
        elif count > 28_000:
            targets[label] = 30_000
        elif count >= 18_000:
            targets[label] = 20_000
        else:
            targets[label] = 10_000

    return targets


def balance_dataset(df):
    print("\n" + "=" * 60)
    print("Step 6: Balancing with SMOTE + RandomUnderSampler")
    print("=" * 60)

    X = df.drop(columns=['label']).values.astype(float)
    y = df['label'].values
    counts = Counter(y)

    targets = compute_targets(counts)

    print("\nClass plan:")
    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        t = targets[label]
        direction = "DOWN" if count > t else ("UP" if count < t else "OK")
        print(f"  {direction:<4}  {label:<40} {count:>8,} → {t:,}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    over_strategy  = {}
    under_strategy = {}

    for label, target in targets.items():
        count = counts[label]
        enc   = le.transform([label])[0]
        if count < target and count >= SMOTE_K + 1:
            over_strategy[enc]  = target
        elif count > target:
            under_strategy[enc] = target

    # SMOTE oversampling
    if over_strategy:
        print(f"\n  Applying SMOTE to {len(over_strategy)} classes...")
        smote = SMOTE(
            sampling_strategy=over_strategy,
            k_neighbors=SMOTE_K,
            random_state=RANDOM_SEED
        )
        X, y_enc = smote.fit_resample(X, y_enc)
        print(f"  After SMOTE: {len(X):,} samples")

    # Random undersampling
    if under_strategy:
        current = Counter(y_enc)
        under_final = {k: v for k, v in under_strategy.items()
                       if current.get(k, 0) > v}
        if under_final:
            print(f"  Applying undersampling to {len(under_final)} classes...")
            rus = RandomUnderSampler(
                sampling_strategy=under_final,
                random_state=RANDOM_SEED
            )
            X, y_enc = rus.fit_resample(X, y_enc)
            print(f"  After undersampling: {len(X):,} samples")

    y_out    = le.inverse_transform(y_enc)
    feat_cols = df.drop(columns=['label']).columns
    df_out   = pd.DataFrame(X, columns=feat_cols)
    df_out['label'] = y_out
    return df_out


# ── Step 7: Save with subfolder structure ─────────────────────────────────────
def save_by_subfolder(df):
    print("\n" + "=" * 60)
    print("Step 7: Saving to balanced_dataset/ subfolders")
    print("=" * 60)

    for label in df['label'].unique():
        subfolder = get_subfolder(label)
        out_dir   = os.path.join(OUTPUT_DIR, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        subset = df[df['label'] == label].sort_values('label').reset_index(drop=True)
        out    = os.path.join(out_dir, f"{label}.csv")
        subset.to_csv(out, index=False)
        print(f"  {label:<40} {len(subset):>8,} rows → {subfolder}/")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset()
    df = drop_metadata(df)
    df = remove_duplicates(df)
    df = remove_zero_rows(df)
    df, removed_features = remove_constant_features(df)
    df = balance_dataset(df)
    save_by_subfolder(df)

    print("\n" + "=" * 60)
    print("Final class distribution:")
    print("=" * 60)
    for label, count in sorted(df['label'].value_counts().items()):
        print(f"  {label:<40} {count:>8,}")

    print(f"\nTotal rows:    {len(df):,}")
    print(f"Total features: {df.shape[1] - 1}")
    print(f"Output:         {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
