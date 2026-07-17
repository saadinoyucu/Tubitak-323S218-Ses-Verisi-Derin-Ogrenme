"""
generate_all_graphs.py
----------------------
Tüm grafikleri tek bir script ile oluşturur ve new_graph/ klasörüne kaydeder.

Birleştirilen scriptler:
  1. plot_metrics_per_model_train.py   → Model metrikleri bar chart
  2. label_distribution                 → Train + test label dağılımı
  3. run_report_utils_grafikler.py     → Heatmap + model karşılaştırma bar chart
  4. run_training_plots_grafikler.py   → Eğitim eğrileri (by_run, by_model, summary)
  5. tsne_visualize.py                 → T-SNE görselleştirmesi

Kullanım:
  python generate_all_graphs.py                   # Tüm grafikleri oluşturur
  python generate_all_graphs.py --skip-tsne       # T-SNE hariç (hızlı mod)
  python generate_all_graphs.py --only tsne       # Sadece T-SNE
  python generate_all_graphs.py --max-samples 20000   # T-SNE örneklem limiti
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (  # noqa: E402
    get_canonical_feature_dir,
    get_feature_display_name,
    get_graphs_dir,
    get_results_dir,
    get_results_feature_dir,
    get_results_summary_csv_path,
    list_feature_names,
    normalize_feature_name,
    NEW_GRAPHS_BASE,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SABITLER                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
BASE_DIR       = NEW_GRAPHS_BASE.parent
FEATURE_DIR    = BASE_DIR
DATA_DIR       = REPO_ROOT / "Split_Data"
GRAPHS_DIR     = NEW_GRAPHS_BASE
RESULTS_DIR    = BASE_DIR / "results"
CSV_PATH       = RESULTS_DIR / "summary" / "all_models_comparison.csv"
FEATURE_NAME   = "Unknown"

def set_feature_globals(feature_type: str):
    global BASE_DIR, FEATURE_DIR, DATA_DIR, GRAPHS_DIR, RESULTS_DIR, CSV_PATH, FEATURE_NAME
    feature_key = normalize_feature_name(feature_type)
    FEATURE_DIR = get_results_feature_dir(feature_key)
    BASE_DIR = FEATURE_DIR.parent
    DATA_DIR = get_canonical_feature_dir(feature_key)
    GRAPHS_DIR = get_graphs_dir(feature_key)
    RESULTS_DIR = get_results_dir(feature_key)
    CSV_PATH = get_results_summary_csv_path(feature_key)
    FEATURE_NAME = get_feature_display_name(feature_key)


def _initialize_default_globals() -> None:
    feature_names = list_feature_names()
    if feature_names:
        set_feature_globals(feature_names[0])


_initialize_default_globals()

# T-SNE defaults
LABEL_COL           = "label"
LABEL_MAP           = {"kontrol": 0, "hasta": 1}
TSNE_BATCH_SIZE     = 20_000
DEFAULT_MAX_SAMPLES = 40_000
RANDOM_SEED         = 42


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  YARDIMCI FONKSİYONLAR                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _set_readable_style() -> None:
    """Seaborn + matplotlib için daha okunabilir stil."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FCFCFC",
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def _add_bar_labels(
    ax: plt.Axes,
    fmt: str = "%.2f",
    padding: int = 3,
    fontsize: int = 8,
    rotation: int = 0,
) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding,
                     fontsize=fontsize, rotation=rotation)


def _safe_path_part(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "unknown"
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "unknown"


def _safe_filename(text: str) -> str:
    return _safe_path_part(text)


def _metric_label(metric: str) -> str:
    mapping = {
        "accuracy": "Accuracy",
        "f1": "F1-Score",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
    }
    return mapping.get(metric, metric)


def _format_feature_label(feature_extractor: str | None) -> str:
    return feature_extractor if feature_extractor else "Unknown"


def _detect_feature_extractor() -> str | None:
    if FEATURE_NAME != "Unknown":
        return FEATURE_NAME
    candidates = [
        ("MFCC", ["mfcc"]),
        ("HuBERT", ["hubert"]),
        ("Wav2Vec", ["wav2vec", "wav2vec2", "wav2"]),
        ("google-hear", ["google-hear", "google_hear", "hear"]),
    ]
    haystacks = [
        str(FEATURE_DIR.name).lower(),
        str(RESULTS_DIR.name).lower(),
        str(RESULTS_DIR).lower(),
    ]
    for label, tokens in candidates:
        if any(any(tok in h for tok in tokens) for h in haystacks):
            return label
    return None


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def _spread_ys(ys: list[float], y_min: float, y_max: float, min_gap: float) -> list[float]:
    if not ys:
        return []
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    adjusted = ys[:]
    prev = None
    for idx in order:
        y = adjusted[idx]
        if prev is not None and y < prev + min_gap:
            y = prev + min_gap
        adjusted[idx] = y
        prev = y
    overflow = max(0.0, max(adjusted) - y_max)
    if overflow > 0:
        adjusted = [y - overflow for y in adjusted]
    underflow = max(0.0, y_min - min(adjusted))
    if underflow > 0:
        adjusted = [y + underflow for y in adjusted]
    return adjusted


def _label_line_ends(
    ax: plt.Axes,
    x_last: float,
    y_last_values: list[float],
    labels: list[str],
    colors: list,
) -> None:
    if not y_last_values:
        return

    def y_data_to_axes(y_data: float) -> float:
        _, y_disp = ax.transData.transform((x_last, y_data))
        _, y_axes = ax.transAxes.inverted().transform((0.0, y_disp))
        return float(y_axes)

    y_axes_values = [y_data_to_axes(float(y)) for y in y_last_values]
    count = len(y_axes_values)
    min_gap = 0.06 if count <= 8 else 0.045 if count <= 12 else 0.035
    fontsize = 10 if count <= 8 else 9 if count <= 12 else 8
    y_axes_adjusted = _spread_ys(y_axes_values, y_min=0.03, y_max=0.97, min_gap=min_gap)

    for y_data, y_axes_adj, label, color in zip(
        y_last_values, y_axes_adjusted, labels, colors, strict=False
    ):
        y_data = float(y_data)
        ax.plot([x_last], [y_data], marker="o", markersize=4.5, color=color, zorder=3)
        ax.annotate(
            label,
            xy=(x_last, y_data),
            xycoords="data",
            xytext=(1.02, y_axes_adj),
            textcoords=ax.transAxes,
            va="center", ha="left",
            fontsize=fontsize, color=color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#E0E0E0", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0, "alpha": 0.55},
            clip_on=False, zorder=5,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1) MODEL METRİKLERİ BAR CHART                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def plot_metrics_per_model(output_root: Path | None = None):
    """
    results/summary/all_models_comparison.csv → her model × train_scenario
    kombinasyonu için tüm metrikleri bar-chart olarak çizer.
    Çıktı: new_graph/model_metrics/
    """
    output_root = GRAPHS_DIR if output_root is None else output_root

    if not CSV_PATH.exists():
        print(f"  [metrics] CSV bulunamadı: {CSV_PATH}")
        return

    _set_readable_style()
    df = pd.read_csv(CSV_PATH)
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    metric_label_map = {
        "accuracy": "Accuracy",
        "f1": "F1-Score",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
    }
    metric_labels = [metric_label_map[m] for m in metrics]

    out_dir = output_root / "model_metrics"
    bar_dir = out_dir / "bars"
    bar_dir.mkdir(parents=True, exist_ok=True)

    for model in df["model"].unique():
        for train_set in df["train_scenario"].unique():
            subset = df[(df["model"] == model) & (df["train_scenario"] == train_set)]
            if subset.empty:
                continue

            melted = subset.melt(
                id_vars=["test_dataset"], value_vars=metrics,
                var_name="Metric", value_name="Score",
            )
            melted["Metric"] = melted["Metric"].map(metric_label_map)
            melted["Score"] = melted["Score"].astype(float)

            test_order = sorted(melted["test_dataset"].unique().tolist())
            melted["Metric"] = pd.Categorical(melted["Metric"], categories=metric_labels, ordered=True)

            fig_w = max(10.0, len(test_order) * 2.2)
            fig, ax = plt.subplots(figsize=(fig_w, 6.5))
            sns.barplot(
                data=melted,
                x="test_dataset",
                y="Score",
                hue="Metric",
                order=test_order,
                hue_order=metric_labels,
                palette="Set2",
                edgecolor="white",
                ax=ax,
            )
            ax.set_title(
                f"Model Performance by Test Dataset\n"
                f"Model: {model} | Trained on: {train_set} | Features: {FEATURE_NAME}",
                fontsize=15,
                fontweight="bold",
                pad=16,
            )
            ax.set_xlabel("Evaluated on (Test Dataset)")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.18)
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Metric", bbox_to_anchor=(1.02, 1.0),
                      loc="upper left", frameon=True, borderaxespad=0)
            ax.grid(axis="y", alpha=0.35)
            _add_bar_labels(ax, fmt="%.2f", padding=3, fontsize=8, rotation=0)
            plt.tight_layout()

            fig_name = f"{model}_trained_on_{train_set}_metrics_bars.png"
            fig_path = bar_dir / fig_name
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"    Kaydedildi: {fig_path}")

    print(f"  [metrics] Tamamlandı → {out_dir.resolve()}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2) LABEL DAĞILIMI (Train & Test)                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _load_label_counts(split: str) -> pd.DataFrame | None:
    """Load class counts once per dataset for the requested split."""
    pattern = str(DATA_DIR / f"*_{split}.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"  [{split}_dist] Dosya bulunamadı: {pattern}")
        return None

    dataset_counts = {}
    seen_datasets: set[str] = set()
    for f in files:
        try:
            parsed = _parse_csv_filename(os.path.basename(f))
            if parsed is None:
                continue
            dataset_name, _, _ = parsed
            if dataset_name in seen_datasets:
                continue

            df = pd.read_csv(f)
            if 'label' not in df.columns:
                continue
            labels = df["label"].astype(str).str.strip().str.lower()
            mapped = labels.map(LABEL_MAP)
            if mapped.isna().any():
                mapped = pd.to_numeric(labels, errors="coerce")
            mapped = mapped.dropna()
            counts = mapped.value_counts()
            dataset_counts[dataset_name] = counts
            seen_datasets.add(dataset_name)
        except Exception as e:
            print(f"    HATA: {f} → {e}")

    if not dataset_counts:
        print(f"  [{split}_dist] Hiç label verisi bulunamadı.")
        return None

    df_counts = pd.DataFrame(dataset_counts).fillna(0).T
    df_counts = df_counts.reindex(columns=[0, 1]).fillna(0)
    df_counts.columns = [0, 1]
    return df_counts.astype(int).sort_index()


def plot_label_distribution(output_root: Path | None = None):
    """Plot train and test label counts together with numeric class labels."""
    _ = GRAPHS_DIR if output_root is None else output_root
    shared_root = NEW_GRAPHS_BASE / "label_distribution"
    train_counts = _load_label_counts("train")
    test_counts = _load_label_counts("test")
    if train_counts is None or test_counts is None:
        return

    datasets = train_counts.index.union(test_counts.index).sort_values()
    train_counts = train_counts.reindex(datasets, fill_value=0)
    test_counts = test_counts.reindex(datasets, fill_value=0)
    plot_data = pd.DataFrame(
        {
            "Eğitim - Kontrol (0)": train_counts[0],
            "Eğitim - Hasta (1)": train_counts[1],
            "Test - Kontrol (0)": test_counts[0],
            "Test - Hasta (1)": test_counts[1],
        },
        index=datasets,
    )

    _set_readable_style()
    shared_root.mkdir(parents=True, exist_ok=True)
    fig_w = max(10.5, len(plot_data) * 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.8))
    plot_data.plot(
        kind="bar",
        ax=ax,
        color=["#1B9E77", "#66C2A5", "#D95F02", "#FC8D62"],
        edgecolor="white",
        width=0.78,
    )
    ax.set_title(
        "Eğitim ve Test Veri Setlerinde Sınıf Dağılımı",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Veri Seti")
    ax.set_ylabel("Örnek Sayısı")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=0)
    max_count = int(plot_data.to_numpy().max())
    ax.set_ylim(0, max_count * 1.15 if max_count else 1)
    for container in ax.containers:
        labels = [f"{int(v)}" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=10)
    ax.legend(title="Bölüm - Sınıf", ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=True)

    plt.tight_layout()
    out_path = shared_root / "train_test_class_distribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [label_dist] Kaydedildi: {out_path}")


def plot_train_label_distribution(output_root: Path | None = None):
    plot_label_distribution(output_root)


def plot_test_label_distribution(output_root: Path | None = None):
    plot_label_distribution(output_root)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3) REPORT UTILS GRAFİKLERİ (Heatmap + Model Comparison Bar)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _ordered_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index=["model", "train_scenario"],
            columns="test_dataset",
            values=metric,
            aggfunc="mean",
        )
        .astype(float)
        .sort_index(axis=0, level=[0, 1], ascending=True)
    )
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot.index = [f"{m} | {t}" for (m, t) in pivot.index.to_list()]
    return pivot


def _draw_heatmap_readable(
    pivot: pd.DataFrame,
    title: str,
    save_path: Path,
    dpi: int = 250,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    pivot = pivot.copy()
    if not pivot.empty:
        row_order = pivot.mean(axis=1).sort_values(ascending=False).index
        pivot = pivot.loc[row_order]
    n_rows, n_cols = pivot.shape
    fig_w = max(10.0, n_cols * 2.0)
    fig_h = max(5.0, n_rows * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlGnBu",
        vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": title, "shrink": 0.8}, ax=ax,
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("Test Dataset")
    ax.set_ylabel("Model | Train Dataset")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    if n_rows and n_cols:
        overall_mean = float(np.nanmean(pivot.to_numpy()))
        ax.text(
            0.99, 1.02, f"Overall mean: {overall_mean:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#E0E0E0", "alpha": 0.9},
        )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_model_comparison_readable(
    df: pd.DataFrame,
    metrics: list[str],
    save_path: Path,
    dpi: int = 250,
    feature_extractor: str | None = None,
) -> None:
    agg = df.groupby("model")[metrics].mean().reset_index()
    if "accuracy" in metrics:
        agg = agg.sort_values("accuracy", ascending=False)
    else:
        agg = agg.sort_values(metrics[0], ascending=False)
    best_model = agg.iloc[0]["model"] if not agg.empty else "N/A"

    long_df = agg.melt(id_vars=["model"], var_name="metric", value_name="score")
    long_df["metric"] = long_df["metric"].map(_metric_label)

    fig_w = max(10.0, len(agg) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    palette = sns.color_palette("tab10", n_colors=len(metrics))
    sns.barplot(data=long_df, x="model", y="score", hue="metric",
                palette=palette, ax=ax, edgecolor="white")

    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Model")
    ax.set_ylabel("Skor")
    title = "Modellere Göre Ortalama Başarı Metrikleri"
    if feature_extractor:
        title += f" - Öznitelikler: {feature_extractor}"
    ax.set_title(title)
    ax.legend(title="Metrik", bbox_to_anchor=(1.02, 1.0),
              loc="upper left", frameon=True, borderaxespad=0)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    _add_bar_labels(ax, fmt="%.2f", padding=3, fontsize=8, rotation=0)
    ax.text(
        0.01, 0.02, f"En iyi model (ortalamaya göre): {best_model}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E0E0E0", "alpha": 0.9},
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_report_graphs(output_root: Path | None = None, dpi: int = 250):
    """
    report_utils modülünü kullanarak:
      - all_models_comparison.csv
      - Heatmap (her metrik için)
      - Model karşılaştırma bar chart
    oluşturur.
    """
    try:
        import report_utils
    except ImportError:
        print("  [report] report_utils modülü bulunamadı, atlanıyor.")
        return

    _set_readable_style()
    feature_extractor = _detect_feature_extractor()
    output_root = GRAPHS_DIR if output_root is None else output_root

    df = report_utils.collect_all_results(str(RESULTS_DIR))
    if df.empty:
        print("  [report] Hiç sonuç dosyası bulunamadı.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["feature_extractor"] = _format_feature_label(feature_extractor)

    # CSV kaydet
    csv_path = output_root / "all_models_comparison.csv"
    report_utils.save_summary_csv(df, str(csv_path))
    print(f"    CSV kaydedildi: {csv_path}")

    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print("  [report] Uygun metrik kolonu bulunamadı.")
        return

    # Tek grafik: tüm metrikleri aynı grafikte göster
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    agg = df.groupby("model")[available_metrics].mean().reset_index()
    if "accuracy" in available_metrics:
        agg = agg.sort_values("accuracy", ascending=False)
    else:
        agg = agg.sort_values(available_metrics[0], ascending=False)

    long_df = agg.melt(id_vars=["model"], var_name="metric", value_name="score")
    long_df["metric"] = long_df["metric"].map(_metric_label)

    fig_w = max(10.0, len(agg) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    sns.barplot(
        data=long_df,
        x="model",
        y="score",
        hue="metric",
        palette="Set2",
        edgecolor="white",
        ax=ax,
    )
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    title = "Overall Metrics by Model (Mean Across Datasets)"
    if feature_extractor:
        title += f" — Features: {feature_extractor}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1.0),
              loc="upper left", frameon=True, borderaxespad=0)
    _add_bar_labels(ax, fmt="%.2f", padding=3, fontsize=8, rotation=0)
    plt.tight_layout()
    summary_path = summary_dir / "overall_metrics_bar.png"
    fig.savefig(summary_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grafik kaydedildi: {summary_path}")

    # Model karşılaştırma bar chart
    bar_path = output_root / "model_comparison_bar.png"
    _draw_model_comparison_readable(
        df, available_metrics, bar_path, dpi=dpi,
        feature_extractor=feature_extractor,
    )
    print(f"    Bar grafik kaydedildi: {bar_path}")
    print(f"  [report] Tamamlandi -> {output_root.resolve()}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4) EĞİTİM EĞRİLERİ (Training Curves / Overlay / Heatmap)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True)
class RunKey:
    model: str
    train_dataset: str


def _discover_runs(results_root: Path) -> dict[RunKey, Path]:
    runs: dict[RunKey, Path] = {}
    for history_path in results_root.glob("*/train_*/train_history.csv"):
        try:
            train_dir = history_path.parent
            model_dir = train_dir.parent
            model = model_dir.name
            train_dataset = train_dir.name.replace("train_", "", 1)
            runs[RunKey(model=model, train_dataset=train_dataset)] = history_path
        except Exception:
            continue
    return runs


def _load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("epoch", "train_loss", "train_acc", "lr"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df.sort_values("epoch")


def _summarize_history(df: pd.DataFrame) -> dict[str, float | int]:
    epochs = int(df["epoch"].max())
    final_loss = float(df["train_loss"].iloc[-1])
    final_acc = float(df["train_acc"].iloc[-1])
    best_loss_idx = int(df["train_loss"].astype(float).idxmin())
    best_acc_idx = int(df["train_acc"].astype(float).idxmax())
    best_loss = float(df.loc[best_loss_idx, "train_loss"])
    best_acc = float(df.loc[best_acc_idx, "train_acc"])
    epoch_best_loss = int(df.loc[best_loss_idx, "epoch"])
    epoch_best_acc = int(df.loc[best_acc_idx, "epoch"])
    start_lr = float(df["lr"].iloc[0])
    end_lr = float(df["lr"].iloc[-1])
    return {
        "epochs": epochs,
        "final_train_loss": final_loss, "final_train_acc": final_acc,
        "best_train_loss": best_loss, "best_train_acc": best_acc,
        "epoch_best_loss": epoch_best_loss, "epoch_best_acc": epoch_best_acc,
        "start_lr": start_lr, "end_lr": end_lr,
    }


def _plot_run_curves(
    df: pd.DataFrame, key: RunKey, feature_extractor: str | None,
    save_path: Path, dpi: int = 250, smooth_window: int = 5,
) -> None:
    epochs = df["epoch"].to_numpy()
    loss = df["train_loss"].to_numpy(dtype=float)
    acc = df["train_acc"].to_numpy(dtype=float)

    loss_s = _rolling_mean(loss, smooth_window)
    acc_s = _rolling_mean(acc, smooth_window)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True)

    ax = axes[0]
    ax.plot(epochs, loss_s, color="#E53935", lw=2.0, label="Train loss")
    ax.set_ylabel("Loss")
    title = f"Training Curves — {key.model} / {key.train_dataset}"
    if feature_extractor:
        title += f" — {feature_extractor}"
    ax.set_title(title, pad=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.plot(epochs, acc_s, color="#1E88E5", lw=2.0, label="Train accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_model_overlay(
    histories: dict[str, pd.DataFrame], model: str,
    feature_extractor: str | None, save_path: Path,
    dpi: int = 250, smooth_window: int = 5,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    palette = sns.color_palette("tab10", n_colors=max(3, len(histories)))

    for i, (train_dataset, df) in enumerate(sorted(histories.items())):
        epochs = df["epoch"].to_numpy()
        loss = _rolling_mean(df["train_loss"].to_numpy(dtype=float), smooth_window)
        acc = _rolling_mean(df["train_acc"].to_numpy(dtype=float), smooth_window)
        color = palette[i % len(palette)]
        label = f"Trained on: {train_dataset}"
        axes[0].plot(epochs, loss, lw=2.2, label=label, color=color)
        axes[1].plot(epochs, acc, lw=2.2, label=label, color=color)

    title = f"Training Curves Overlay — Model: {model}"
    if feature_extractor:
        title += f" — Features: {feature_extractor}"
    axes[0].set_title(title, pad=12)
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.2)
    axes[0].legend(title="Train Dataset", ncol=1, frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_overlay(
    histories: dict[str, pd.DataFrame], train_dataset: str,
    feature_extractor: str | None, save_path: Path,
    dpi: int = 250, smooth_window: int = 5,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    palette = sns.color_palette("tab10", n_colors=max(3, len(histories)))

    for i, (model_name, df) in enumerate(sorted(histories.items())):
        epochs = df["epoch"].to_numpy()
        loss = _rolling_mean(df["train_loss"].to_numpy(dtype=float), smooth_window)
        acc = _rolling_mean(df["train_acc"].to_numpy(dtype=float), smooth_window)
        color = palette[i % len(palette)]
        label = f"Model: {model_name}"
        axes[0].plot(epochs, loss, lw=2.2, label=label, color=color)
        axes[1].plot(epochs, acc, lw=2.2, label=label, color=color)

    title = f"Training Curves Overlay — Train Dataset: {train_dataset}"
    if feature_extractor:
        title += f" — Features: {feature_extractor}"
    axes[0].set_title(title, pad=12)
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.2)
    axes[0].legend(title="Models", ncol=1, frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_training_heatmap(
    pivot: pd.DataFrame, title: str, save_path: Path,
    dpi: int = 250, cmap: str = "YlGnBu",
    vmin: float | None = None, vmax: float | None = None,
    fmt: str = ".2f",
) -> None:
    n_rows, n_cols = pivot.shape
    fig_w = max(9.0, n_cols * 2.2)
    fig_h = max(4.5, n_rows * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot.astype(float), annot=True, fmt=fmt, cmap=cmap,
        vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": title, "shrink": 0.8}, ax=ax,
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("Train Dataset")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_training_comparison_bars(
    summary_df: pd.DataFrame, summary_dir: Path,
    feature_extractor: str | None, dpi: int = 250,
) -> None:
    def _bar_simple(ax: plt.Axes, data: pd.DataFrame, title: str, y_label: str, value_col: str):
        ax.bar(data["label"], data[value_col], color="#5C6BC0", edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(y_label)
        ax.set_ylim(0, 1.18 if "acc" in value_col else max(0.1, float(data[value_col].max()) * 1.20))
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8, rotation=0)

    # Final accuracy by model
    by_model = (
        summary_df.groupby("model")["final_train_acc"]
        .mean().reset_index()
        .rename(columns={"model": "label"})
        .sort_values("final_train_acc", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(max(8.0, len(by_model) * 1.6), 5.8))
    title = "Final Train Accuracy by Model"
    if feature_extractor:
        title += f" — Features: {feature_extractor}"
    _bar_simple(ax, by_model, title, "Accuracy", "final_train_acc")
    plt.tight_layout()
    out_path = summary_dir / "final_train_acc_by_model.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Final accuracy by train dataset
    by_ds = (
        summary_df.groupby("train_dataset")["final_train_acc"]
        .mean().reset_index()
        .rename(columns={"train_dataset": "label"})
        .sort_values("final_train_acc", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(max(8.0, len(by_ds) * 1.6), 5.8))
    title = "Final Train Accuracy by Train Dataset"
    if feature_extractor:
        title += f" — Features: {feature_extractor}"
    _bar_simple(ax, by_ds, title, "Accuracy", "final_train_acc")
    plt.tight_layout()
    out_path = summary_dir / "final_train_acc_by_dataset.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    output_root: Path | None = None, dpi: int = 250, smooth_window: int = 5,
):
    """
    results/ altındaki train_history.csv dosyalarını tarayıp:
      - by_run/   → Her model/dataset için eğitim eğrisi
      - by_model/ → Model bazında overlay
      - summary/  → Heatmap + CSV özet
    oluşturur.
    Çıktı: new_graph/training/
    """
    _set_readable_style()
    feature_extractor = _detect_feature_extractor()
    output_root = GRAPHS_DIR if output_root is None else output_root
    out_root = output_root / "training"

    runs = _discover_runs(RESULTS_DIR)
    if not runs:
        print("  [training] train_history.csv dosyası bulunamadı.")
        return

    summary_rows = []
    by_model: dict[str, dict[str, pd.DataFrame]] = {}
    by_dataset: dict[str, dict[str, pd.DataFrame]] = {}

    for key in sorted(runs.keys(), key=lambda k: (k.model, k.train_dataset)):
        history_path = runs[key]
        df = _load_history(history_path)
        stats = _summarize_history(df)
        summary_rows.append({
            "model": key.model,
            "train_dataset": key.train_dataset,
            "feature_extractor": feature_extractor or "",
            **stats,
            "history_path": str(history_path),
        })
        by_model.setdefault(key.model, {})[key.train_dataset] = df
        by_dataset.setdefault(key.train_dataset, {})[key.model] = df

        run_plot_path = (
            out_root / "by_run" / _safe_path_part(key.model)
            / _safe_path_part(key.train_dataset) / "train_curves.png"
        )
        _plot_run_curves(df, key, feature_extractor, run_plot_path,
                         dpi=dpi, smooth_window=smooth_window)

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "train_dataset"])
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "train_history_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"    Summary CSV: {summary_csv}")

    # Model overlay (trained on different datasets)
    for model, histories in sorted(by_model.items()):
        model_plot = out_root / "by_model" / _safe_path_part(model) / "train_curves_overlay.png"
        _plot_model_overlay(
            histories, model=model, feature_extractor=feature_extractor,
            save_path=model_plot, dpi=dpi, smooth_window=smooth_window,
        )

    # Dataset overlay (different models on same dataset)
    for train_dataset, histories in sorted(by_dataset.items()):
        dataset_plot = out_root / "by_dataset" / _safe_path_part(train_dataset) / "train_curves_overlay.png"
        _plot_dataset_overlay(
            histories, train_dataset=train_dataset, feature_extractor=feature_extractor,
            save_path=dataset_plot, dpi=dpi, smooth_window=smooth_window,
        )

    # Comparative academic-style summary bars
    _plot_training_comparison_bars(
        summary_df, summary_dir=summary_dir,
        feature_extractor=feature_extractor, dpi=dpi,
    )

    print(f"  [training] Tamamlandı → {out_root.resolve()}")


# ╔════════════════════════════════════════════════════════════════════════════
#  5) FEATURE EXTRACTOR KARŞILAŞTIRMA (MODEL BAZLI)
# ╚════════════════════════════════════════════════════════════════════════════
def plot_feature_comparison(output_root: Path | None = None, dpi: int = 250):
    """
    Farklı feature extractor'ları aynı model altında karşılaştırır.
    Çıktı: new_graph/feature_comparison/by_model/
    """
    _set_readable_style()
    output_root = GRAPHS_DIR if output_root is None else output_root
    shared_root = NEW_GRAPHS_BASE / "feature_comparison"
    out_root = shared_root
    out_root.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    feature_rows = []

    # Her feature'ın summary CSV'sini oku
    for feature_name in list_feature_names():
        feature_key = normalize_feature_name(feature_name)
        csv_path = get_results_summary_csv_path(feature_key)
        if not csv_path.exists():
            print(f"  [feature] CSV bulunamadı: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  [feature] CSV okunamadı: {csv_path} → {exc}")
            continue
        df["feature_extractor"] = get_feature_display_name(feature_key)
        feature_rows.append(df)

    if not feature_rows:
        print("  [feature] Karşılaştırma için veri bulunamadı.")
        return

    all_df = pd.concat(feature_rows, ignore_index=True)
    available_metrics = [m for m in metrics if m in all_df.columns]
    if not available_metrics:
        print("  [feature] Uygun metrik kolonu bulunamadı.")
        return

    # Model bazlı karşılaştırma
    for model_name in sorted(all_df["model"].unique().tolist()):
        model_df = all_df[all_df["model"] == model_name]
        if model_df.empty:
            continue

        agg = (
            model_df.groupby("feature_extractor")[available_metrics]
            .mean()
            .reset_index()
        )
        if "accuracy" in available_metrics:
            agg = agg.sort_values("accuracy", ascending=False)
        else:
            agg = agg.sort_values(available_metrics[0], ascending=False)

        long_df = agg.melt(
            id_vars=["feature_extractor"],
            var_name="metric",
            value_name="score",
        )
        long_df["metric"] = long_df["metric"].map(_metric_label)

        fig_w = max(9.0, len(agg) * 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, 6.2))
        sns.barplot(
            data=long_df,
            x="feature_extractor",
            y="score",
            hue="metric",
            palette="Set2",
            edgecolor="white",
            ax=ax,
        )
        ax.set_ylim(0, 1.18)
        ax.set_xlabel("Feature Extractor")
        ax.set_ylabel("Score")
        ax.set_title(
            f"Feature Extractor Comparison — Model: {model_name}",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Metric", bbox_to_anchor=(1.02, 1.0),
                  loc="upper left", frameon=True, borderaxespad=0)
        _add_bar_labels(ax, fmt="%.2f", padding=3, fontsize=8, rotation=0)

        best_feature = agg.iloc[0]["feature_extractor"] if not agg.empty else "N/A"
        ax.text(
            0.01, 0.02, f"Best (by mean): {best_feature}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E0E0E0", "alpha": 0.9},
        )

        plt.tight_layout()
        model_dir = out_root / "by_model" / _safe_path_part(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        bar_path = model_dir / "feature_comparison_bar.png"
        fig.savefig(bar_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"    Kaydedildi: {bar_path}")

        # Heatmap: feature × metric
        pivot = agg.set_index("feature_extractor")[available_metrics]
        pivot.index = [str(i) for i in pivot.index]
        pivot.columns = [_metric_label(m) for m in pivot.columns]

        fig, ax = plt.subplots(figsize=(max(7.0, len(available_metrics) * 1.6),
                                        max(4.5, len(pivot) * 0.8)))
        sns.heatmap(
            pivot.astype(float),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Score", "shrink": 0.8},
            ax=ax,
        )
        ax.set_title(
            f"Feature Extractor Heatmap — Model: {model_name}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Metric")
        ax.set_ylabel("Feature Extractor")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        plt.tight_layout()
        heat_path = model_dir / "feature_comparison_heatmap.png"
        fig.savefig(heat_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"    Kaydedildi: {heat_path}")

    # Genel (model bağımsız) özet grafikler
    overall_dir = out_root / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)

    overall = (
        all_df.groupby("feature_extractor")[available_metrics]
        .mean()
        .reset_index()
    )
    if "accuracy" in available_metrics:
        overall = overall.sort_values("accuracy", ascending=False)
    else:
        overall = overall.sort_values(available_metrics[0], ascending=False)

    long_overall = overall.melt(
        id_vars=["feature_extractor"],
        var_name="metric",
        value_name="score",
    )
    long_overall["metric"] = long_overall["metric"].map(_metric_label)

    fig_w = max(9.0, len(overall) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    sns.barplot(
        data=long_overall,
        x="feature_extractor",
        y="score",
        hue="metric",
        palette="Set2",
        edgecolor="white",
        ax=ax,
    )
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Öznitelik Çıkarıcı")
    ax.set_ylabel("Skor")
    ax.set_title(
        "Öznitelik Çıkarıcıların Genel Karşılaştırması\n"
        "(Modeller ve Veri Setleri Üzerindeki Ortalama Performans)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Metrik", bbox_to_anchor=(1.02, 1.0),
              loc="upper left", frameon=True, borderaxespad=0)
    _add_bar_labels(ax, fmt="%.2f", padding=3, fontsize=8, rotation=0)
    best_feature = overall.iloc[0]["feature_extractor"] if not overall.empty else "N/A"
    ax.text(
        0.01, 0.02, f"En iyi (ortalamaya göre): {best_feature}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E0E0E0", "alpha": 0.9},
    )
    plt.tight_layout()
    overall_bar = overall_dir / "feature_overall_bar.png"
    fig.savefig(overall_bar, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Kaydedildi: {overall_bar}")

    # Accuracy odaklı: Model x Feature heatmap
    if "accuracy" in available_metrics:
        heat_pivot = (
            all_df.pivot_table(
                index="model",
                columns="feature_extractor",
                values="accuracy",
                aggfunc="mean",
            )
            .reindex(index=sorted(all_df["model"].unique().tolist()))
        )
        fig, ax = plt.subplots(figsize=(max(7.0, heat_pivot.shape[1] * 1.6),
                                        max(4.5, heat_pivot.shape[0] * 0.7)))
        sns.heatmap(
            heat_pivot.astype(float),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Accuracy", "shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Accuracy Heatmap (Model × Feature Extractor)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Feature Extractor")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        plt.tight_layout()
        overall_heat = overall_dir / "feature_accuracy_heatmap.png"
        fig.savefig(overall_heat, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"    Kaydedildi: {overall_heat}")

    print(f"  [feature] Tamamlandi -> {out_root.resolve()}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5) T-SNE GÖRSELLEŞTİRMESİ                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _parse_csv_filename(filename: str):
    """dataset_feature_split.csv -> (dataset, split, feature) or None."""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) < 3:
        return None
    split = parts[-1]
    feature = parts[-2]
    dataset = "_".join(parts[:-2])
    if split.lower() not in ("train", "test"):
        return None
    return dataset, split, feature


def _load_csv_features(path: str, max_samples: int | None = None):
    """CSV dosyasından feature matris ve label vektörü okur."""
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    exclude_cols = {LABEL_COL, "file_name", "filename"}
    feature_cols = [c for c in cols if c not in exclude_cols]
    cols_to_read = feature_cols + [LABEL_COL]

    X_parts, y_parts = [], []
    reader = pd.read_csv(path, usecols=cols_to_read, chunksize=TSNE_BATCH_SIZE, low_memory=False)
    for df in reader:
        labels = df[LABEL_COL].astype(str).str.strip().str.lower()
        y = labels.map(LABEL_MAP)
        if y.isna().any():
            y = pd.to_numeric(labels, errors="coerce")
        if y.isna().any():
            bad_vals = labels[y.isna()].unique()[:5]
            raise ValueError(f"Unknown label values: {bad_vals}")
        y = y.values.astype(np.int32)

        df.drop(columns=[LABEL_COL], inplace=True)
        X = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        X_parts.append(X)
        y_parts.append(y)

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    del X_parts, y_parts
    gc.collect()

    # Balanced subsampling
    rng = np.random.RandomState(RANDOM_SEED)
    real_idx = np.where(y_all == 0)[0]
    fake_idx = np.where(y_all == 1)[0]
    per_class = min(len(real_idx), len(fake_idx))
    if max_samples is not None:
        per_class = min(per_class, max_samples // 2)

    real_chosen = rng.choice(real_idx, size=per_class, replace=False)
    fake_chosen = rng.choice(fake_idx, size=per_class, replace=False)
    idx = np.concatenate([real_chosen, fake_chosen])
    rng.shuffle(idx)

    return X_all[idx], y_all[idx]


def _run_tsne_and_plot(
    X: np.ndarray, y: np.ndarray,
    dataset: str, split: str, method: str,
    save_dir: Path, perplexity: int = 30,
) -> Path:
    """T-SNE uygulayıp scatter plot kaydeder."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape
    print(f"    T-SNE  (n={n_samples}, feat={n_features}, perp={perplexity}) ...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    effective_perp = min(perplexity, max(5, n_samples // 4))
    tsne = TSNE(
        n_components=2, perplexity=effective_perp, learning_rate="auto",
        init="pca", random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1,
    )
    X_tsne = tsne.fit_transform(X_scaled)
    del X_scaled
    gc.collect()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {0: "#2196F3", 1: "#F44336"}
    n_control = int((y == 0).sum())
    n_patient = int((y == 1).sum())
    labels_text = {
        0: f"Control (n={n_control})",
        1: f"Patient (n={n_patient})",
    }

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colors[cls], label=labels_text[cls],
            alpha=0.55, s=12, edgecolors="none",
        )

    title = f"T-SNE  —  {dataset}  /  {split.capitalize()}  /  {method.upper()}"
    subtitle = f"(n = {n_samples:,}  |  perplexity = {effective_perp})"
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("T-SNE Component 1", fontsize=11)
    ax.set_ylabel("T-SNE Component 2", fontsize=11)
    ax.legend(fontsize=11, markerscale=2.5, framealpha=0.9, loc="best")
    ax.grid(alpha=0.2)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"tSNE_{dataset}_{split}_{method}.png"
    save_path = save_dir / fname
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ Kaydedildi: {save_path}")

    del X_tsne
    gc.collect()
    return save_path


def plot_tsne(
    output_root: Path | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    perplexity: int = 30,
):
    """
    Tüm csv dosyaları için T-SNE scatter plot oluşturur.
    Çıktı: new_graph/t-SNE/
    """
    output_root = GRAPHS_DIR if output_root is None else output_root
    save_dir = output_root / "t-SNE"
    csv_files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    current_feature_key = normalize_feature_name(FEATURE_NAME) if FEATURE_NAME != "Unknown" else None

    if not csv_files:
        print("  [tsne] CSV dosyası bulunamadı.")
        return

    valid_files = []
    for pf in csv_files:
        parsed = _parse_csv_filename(os.path.basename(pf))
        if parsed is not None:
            dataset, split, method = parsed
            if current_feature_key is not None:
                method_key = normalize_feature_name(method)
                if method_key != current_feature_key:
                    continue
            valid_files.append((pf, dataset, split, method))

    if not valid_files:
        print("  [tsne] X_Y_Z.csv formatına uyan dosya bulunamadı.")
        return

    print(f"    Max samples : {max_samples:,}")
    print(f"    Perplexity  : {perplexity}")
    if current_feature_key is not None:
        print(f"    Feature     : {current_feature_key}")
    print(f"    Dosya sayısı: {len(valid_files)}")

    results = []
    for i, (filepath, dataset, split, method) in enumerate(valid_files, 1):
        fname = os.path.basename(filepath)
        print(f"\n    [{i}/{len(valid_files)}] {fname}")
        t0 = time.time()
        try:
            print(f"    Veri yükleniyor ...")
            X, y = _load_csv_features(filepath, max_samples=max_samples)
            n_real = int((y == 0).sum())
            n_fake = int((y == 1).sum())
            print(f"    Yüklendi: {len(X):,} (Control: {n_real:,}, Patient: {n_fake:,})")
            _run_tsne_and_plot(
                X, y, dataset=dataset, split=split, method=method,
                save_dir=save_dir, perplexity=perplexity,
            )
            elapsed = time.time() - t0
            results.append((fname, "✓", f"{elapsed:.1f}s"))
        except Exception as e:
            elapsed = time.time() - t0
            results.append((fname, "✗", str(e)))
            print(f"    [HATA] {e}")
        gc.collect()

    print(f"\n  [tsne] Özet:")
    for fname, status, info in results:
        print(f"    {status}  {fname:<45s}  {info}")
    print(f"  [tsne] Tamamlandı → {save_dir.resolve()}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ANA ÇALIŞTIRMA                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def generate_all_graphs(
    skip_tsne: bool = False,
    only: str | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    perplexity: int = 30,
    dpi: int = 250,
    smooth: int = 5,
):
    """Tum grafikleri sirayla olusturur."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    steps = {
        "metrics":  ("Model Metrikleri Bar Charts",        lambda: plot_metrics_per_model()),
        "labels":   ("Eğitim ve Test Sınıf Dağılımı",      lambda: plot_label_distribution()),
        "report":   ("Report Graphleri (Heatmap + Bar)",  lambda: plot_report_graphs(dpi=dpi)),
        "training": ("Eğitim Eğrileri (Curves/Overlay)",   lambda: plot_training_curves(dpi=dpi, smooth_window=smooth)),
        "feature":  ("Feature Extractor Karşılaştırma",    lambda: plot_feature_comparison(dpi=dpi)),
        "tsne":     ("T-SNE Görselleştirmesi",             lambda: plot_tsne(max_samples=max_samples, perplexity=perplexity)),
    }

    total = len(steps)
    print(f"\n{'='*65}")
    print(f"  GENERATE ALL GRAPHS")
    print(f"  Graph cikti klasoru : {GRAPHS_DIR.resolve()}")
    print(f"  Feature Extractor    : {FEATURE_NAME}")
    print(f"{'='*65}\n")

    if only in {"train", "test"}:
        only = "labels"

    if only:
        if only not in steps:
            print(f"[HATA] Geçersiz --only değeri: '{only}'")
            print(f"  Geçerli değerler: {', '.join(steps.keys())}")
            return
        print(f">>> [1/1]  {steps[only][0]}")
        steps[only][1]()
        print()
    else:
        for i, (key, (label, fn)) in enumerate(steps.items(), 1):
            if key == "tsne" and skip_tsne:
                print(f">>> [{i}/{total}]  {label}  [SKIPPED --skip-tsne]")
                print()
                continue
            print(f">>> [{i}/{total}]  {label}")
            fn()
            print()

    print(f"{'='*65}")
    print(f"  Tum grafikler basariyla olusturuldu!")
    print(f"  Output dir: {GRAPHS_DIR.resolve()}")
    print(f"{'='*65}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Tum grafikleri tek bir komutla olusturur -> new_graph/",
    )
    parser.add_argument("--feature_type", type=str, required=True,
                        choices=["MFCC", "HuBERT", "Wac2Vec", "Wav2Vec", "google-HeAR", "all"],
                        help="Hangi feature için grafik çizileceği.")
    parser.add_argument("--skip-tsne", action="store_true",
                        help="T-SNE adımını atla (daha hızlı)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["metrics", "labels", "train", "test", "report", "training", "feature", "tsne"],
                        help="Sadece belirli bir adımı çalıştır")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"T-SNE max örneklem sayısı (varsayılan: {DEFAULT_MAX_SAMPLES})")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="T-SNE perplexity parametresi (varsayılan: 30)")
    parser.add_argument("--dpi", type=int, default=250,
                        help="Çıktı görüntü DPI (varsayılan: 250)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Eğitim eğrileri rolling mean penceresi (varsayılan: 5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    features = list_feature_names() if args.feature_type == "all" else [args.feature_type]

    for f in features:
        set_feature_globals(f)
        generate_all_graphs(
            skip_tsne=args.skip_tsne,
            only=args.only,
            max_samples=args.max_samples,
            perplexity=args.perplexity,
            dpi=args.dpi,
            smooth=args.smooth,
        )
