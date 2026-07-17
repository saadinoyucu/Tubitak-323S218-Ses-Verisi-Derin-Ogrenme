"""
dataset_utils.py
----------------
CSV streaming loader, label mapping, StandardScaler and DataLoader helpers.
"""

import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "kontrol": 0,
    "hasta": 1,
}
LABEL_COL = "label"
BATCH_SIZE = 8192  # chunksize for pandas streaming


# ---------------------------------------------------------------------------
# Helper: pos_weight
# ---------------------------------------------------------------------------
def compute_pos_weight(n_neg: int, n_pos: int) -> torch.Tensor:
    """
    pos_weight = #negative / #positive -> BCE loss weight.
    """
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Helpers: column and label handling
# ---------------------------------------------------------------------------
def _infer_usecols(path: str) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in header.columns]
    if LABEL_COL not in cols:
        raise ValueError(f"Label column '{LABEL_COL}' not found in {path}")

    feature_cols = [c for c in cols if c.startswith("emb_") or c.startswith("mfcc_")]
    if not feature_cols:
        feature_cols = [c for c in cols if c not in ["file_name", "filename", LABEL_COL]]

    return [LABEL_COL] + feature_cols


def _map_labels(series: pd.Series) -> np.ndarray:
    norm = series.astype(str).str.strip().str.lower()
    mapped = norm.map(LABEL_MAP)
    if mapped.isna().any():
        mapped = pd.to_numeric(norm, errors="coerce")
    if mapped.isna().any():
        bad_vals = norm[mapped.isna()].unique()[:5]
        raise ValueError(f"Unknown label values: {bad_vals}")
    return mapped.values.astype(np.float32)


# ---------------------------------------------------------------------------
# RAM-Safe Streaming Dataset
# ---------------------------------------------------------------------------
class CsvIterableDataset(IterableDataset):
    """
    Stream CSV from disk in batches to avoid loading all into RAM.
    """

    def __init__(self, path: str, scaler: StandardScaler | None = None, is_train: bool = True):
        self.path = path
        self.scaler = scaler
        self.is_train = is_train
        self.cols = _infer_usecols(path)

    def __iter__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reader = pd.read_csv(self.path, usecols=self.cols, chunksize=BATCH_SIZE, low_memory=False)

        for df_chunk in reader:
            y_chunk = _map_labels(df_chunk[LABEL_COL])
            df_chunk.drop(columns=[LABEL_COL], inplace=True)

            X_chunk = df_chunk.select_dtypes(include=[np.number]).values.astype(np.float32)

            if self.scaler is not None:
                if self.is_train:
                    pass
                X_chunk = self.scaler.transform(X_chunk)

            if X_chunk.ndim == 2:
                X_chunk = X_chunk[:, np.newaxis, :]

            if self.is_train:
                idx = np.random.permutation(len(X_chunk))
                X_chunk = X_chunk[idx]
                y_chunk = y_chunk[idx]

            X_t = torch.tensor(X_chunk, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_chunk, dtype=torch.float32, device=device).unsqueeze(1)

            yield X_t, y_t


# ---------------------------------------------------------------------------
# Pre-scan for pos_weight, input_dim and scaler
# ---------------------------------------------------------------------------
def _scan_dataset(path: str):
    """
    Reads dataset once for scaler training and class counts.
    """
    scaler = StandardScaler()
    n_neg, n_pos, total = 0, 0, 0
    input_dim = None

    cols = _infer_usecols(path)
    reader = pd.read_csv(path, usecols=cols, chunksize=20000, low_memory=False)

    for df_chunk in reader:
        y_chunk = _map_labels(df_chunk[LABEL_COL])
        n_pos += int(y_chunk.sum())
        n_neg += int((y_chunk == 0).sum())

        df_chunk.drop(columns=[LABEL_COL], inplace=True)
        X_chunk = df_chunk.select_dtypes(include=[np.number]).values.astype(np.float32)
        if input_dim is None:
            input_dim = X_chunk.shape[-1]

        scaler.partial_fit(X_chunk)
        total += len(X_chunk)

    return scaler, compute_pos_weight(n_neg, n_pos), input_dim, total, n_pos, n_neg


# ---------------------------------------------------------------------------
# High-level: prepare train dataset
# ---------------------------------------------------------------------------
def prepare_train(train_path: str):
    print(f"[Train] Scan + Scaler fit (Streaming) ... {train_path}")
    scaler, pos_weight, input_dim, total, n_pos, n_neg = _scan_dataset(train_path)

    dataset = CsvIterableDataset(train_path, scaler=scaler, is_train=True)
    loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

    print(f"[Train] {train_path}")
    print(f"  Samples     : {total}")
    print(f"  Pos / Neg   : {n_pos} / {n_neg}")
    print(f"  pos_weight  : {pos_weight.item():.4f}")

    return loader, scaler, pos_weight, input_dim


# ---------------------------------------------------------------------------
# High-level: prepare test dataset
# ---------------------------------------------------------------------------
def prepare_test(test_path: str, scaler: StandardScaler):
    dataset = CsvIterableDataset(test_path, scaler=scaler, is_train=False)
    loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

    with open(test_path, "r", encoding="utf-8") as f:
        total = max(0, sum(1 for _ in f) - 1)
    print(f"[Test]  {test_path}")
    print(f"  Samples     : {total} (Streaming)")

    return loader, None
