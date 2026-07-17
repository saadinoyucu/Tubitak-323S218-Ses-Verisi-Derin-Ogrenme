"""
project_paths.py
----------------
Path helpers for the Split_Data CSV pipeline.
"""

from __future__ import annotations

from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Split_Data"
_RESULTS_BASE_CANDIDATES = [
    BASE_DIR / "Results-And-Graph-2",
    BASE_DIR / "Results-And-Graph",
]
RESULTS_BASE = next((p for p in _RESULTS_BASE_CANDIDATES if p.exists()), _RESULTS_BASE_CANDIDATES[0])
NEW_GRAPHS_BASE = RESULTS_BASE / "new_graph"

FEATURE_DISPLAY = {
    "mfcc": "MFCC",
    "hubert": "HuBERT",
    "wav2vec": "Wav2Vec",
    "googlehear": "google-HeAR",
}

FEATURE_NORMALIZE = {
    "mfcc": "mfcc",
    "hubert": "hubert",
    "hubertbase": "hubert",
    "hubertlarge": "hubert",
    "wav2vec": "wav2vec",
    "wac2vec": "wav2vec",
    "wav2vec2": "wav2vec",
    "googlehear": "googlehear",
    "googlehearv2": "googlehear",
    "google-hear": "googlehear",
    "googlehearbase": "googlehear",
    "googlehearlarge": "googlehear",
}

FILE_RE = re.compile(r"^(?P<dataset>[^_]+)_(?P<feature>[^_]+)_(?P<split>train|test)\.csv$", re.IGNORECASE)


def normalize_feature_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return FEATURE_NORMALIZE.get(key, key)


def get_feature_display_name(feature_key: str) -> str:
    feature_key = normalize_feature_name(feature_key)
    return FEATURE_DISPLAY.get(feature_key, feature_key)


def _scan_split_data():
    datasets = set()
    features = set()
    if not DATA_DIR.exists():
        return datasets, features

    for path in DATA_DIR.glob("*.csv"):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        datasets.add(match.group("dataset"))
        features.add(normalize_feature_name(match.group("feature")))
    return datasets, features


def list_dataset_names() -> list[str]:
    datasets, _ = _scan_split_data()
    return sorted(datasets)


def list_feature_names() -> list[str]:
    _, features = _scan_split_data()
    if not features:
        return ["MFCC", "HuBERT", "Wav2Vec", "google-HeAR"]
    return [get_feature_display_name(f) for f in sorted(features)]


def get_canonical_parquet_path(dataset_name: str, feature_name: str, split: str) -> Path:
    feature_key = normalize_feature_name(feature_name)
    split = split.strip().lower()
    return DATA_DIR / f"{dataset_name}_{feature_key}_{split}.csv"


def get_canonical_feature_dir(feature_key: str) -> Path:
    _ = normalize_feature_name(feature_key)
    return DATA_DIR


def get_results_feature_dir(feature_key: str) -> Path:
    feature_key = normalize_feature_name(feature_key)
    for base in _RESULTS_BASE_CANDIDATES:
        candidate = base / feature_key
        if (candidate / "results").exists():
            return candidate

    for base in _RESULTS_BASE_CANDIDATES:
        candidate = base / feature_key
        if candidate.exists():
            return candidate

    return RESULTS_BASE / feature_key


def get_results_dir(feature_key: str) -> Path:
    return get_results_feature_dir(feature_key) / "results"


def get_graphs_dir(feature_key: str) -> Path:
    feature_key = normalize_feature_name(feature_key)
    feature_dir = get_results_feature_dir(feature_key)
    return feature_dir.parent / "new_graph" / feature_key


def get_results_summary_csv_path(feature_key: str) -> Path:
    return get_results_dir(feature_key) / "summary" / "all_models_comparison.csv"
