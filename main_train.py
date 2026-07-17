from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (  # noqa: E402
    get_canonical_parquet_path,
    get_feature_display_name,
    get_results_dir,
    list_dataset_names,
    list_feature_names,
    normalize_feature_name,
)

from dataset_utils import prepare_test, prepare_train  # noqa: E402
from eval_utils import evaluate_on_testset  # noqa: E402
from model_cnn import CNN_LSTM  # noqa: E402
from model_tcn import Chomp1d, TCN, TemporalBlock  # noqa: F401,E402
from model_tcn_lstm import TCN_LSTM  # noqa: E402
from plot_utils import plot_train_history  # noqa: E402
from report_utils import generate_full_report  # noqa: E402
from train_utils import train_model  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_datasets_dict(feature_type: str):
    normalized_feature = normalize_feature_name(feature_type)
    return {
        dataset_name: {
            "train": str(get_canonical_parquet_path(dataset_name, normalized_feature, "train")),
            "test": str(get_canonical_parquet_path(dataset_name, normalized_feature, "test")),
        }
        for dataset_name in list_dataset_names()
    }


def build_model(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "CNN_LSTM":
        return CNN_LSTM(input_dim=input_dim)
    if model_name == "TCN":
        return TCN(input_dim=input_dim, num_channels=[64, 128, 64], kernel_size=3)
    if model_name == "TCN_LSTM":
        return TCN_LSTM(input_dim=input_dim, tcn_channels=[64, 64], kernel_size=3, lstm_hidden=64)
    raise ValueError(f"Unknown model: {model_name}")


def get_save_dir(results_dir: str, model_name: str, train_ds_name: str) -> str:
    return os.path.join(results_dir, model_name, f"train_{train_ds_name}")


def run_pipeline(feature_type: str):
    normalized_feature = normalize_feature_name(feature_type)
    display_name = get_feature_display_name(normalized_feature)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"\n{'#' * 65}")
    print(f"  Cross-Dataset Deepfake Pipeline - {display_name}")
    print(f"  Device: {DEVICE}")
    print(f"{'#' * 65}\n")

    datasets = get_datasets_dict(normalized_feature)
    results_dir = str(get_results_dir(normalized_feature))
    model_names = ["TCN", "CNN_LSTM", "TCN_LSTM"]

    for model_name in model_names:
        print(f"\n{'=' * 65}")
        print(f"  MODEL: {model_name} (Feature: {display_name})")
        print(f"{'=' * 65}")

        for train_ds_name, paths in datasets.items():
            train_path = paths["train"]
            save_dir = get_save_dir(results_dir, model_name, train_ds_name)

            print(f"\n{'-' * 55}")
            print(f"  TRAIN DATASET: {train_ds_name}")
            print(f"{'-' * 55}")

            try:
                train_loader, scaler, pos_weight, input_dim = prepare_train(train_path)
            except Exception as exc:
                print(f"  [ERROR] Could not load train data ({train_path}): {exc}")
                continue

            model = build_model(model_name, input_dim)

            try:
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    pos_weight=pos_weight,
                    device=DEVICE,
                    save_dir=save_dir,
                )
            except Exception as exc:
                print(f"  [ERROR] Training failed ({model_name}/{train_ds_name}): {exc}")
                continue

            plot_train_history(history, save_dir)

            best_weights_path = os.path.join(save_dir, "model_weights.pt")
            if os.path.exists(best_weights_path):
                model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))
            model.to(DEVICE)

            print(f"\n  Test evaluations (trained on {train_ds_name}):")
            for test_ds_name, test_paths in datasets.items():
                test_path = test_paths["test"]
                try:
                    test_loader, _ = prepare_test(test_path, scaler)
                    evaluate_on_testset(
                        model=model,
                        test_loader=test_loader,
                        test_name=test_ds_name,
                        save_dir=save_dir,
                        device=DEVICE,
                        feature_name=display_name,
                    )
                    del test_loader
                    gc.collect()
                except Exception as exc:
                    print(f"  [ERROR] Test failed ({test_ds_name}): {exc}")

            del model, train_loader, scaler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'=' * 65}")
    print(f"  Building summary report ({display_name})...")
    print(f"{'=' * 65}")
    generate_full_report(results_dir)

    print(f"\n{'#' * 65}")
    print(f"  Pipeline completed. Outputs: {results_dir}")
    print(f"{'#' * 65}\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Pipeline")
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        choices=["MFCC", "HuBERT", "Wac2Vec", "Wav2Vec", "google-HeAR", "all"],
        help="Feature type to train on.",
    )
    args = parser.parse_args()

    feature_names = list_feature_names() if args.feature_type == "all" else [args.feature_type]
    for feature_name in feature_names:
        run_pipeline(feature_name)


if __name__ == "__main__":
    main()
