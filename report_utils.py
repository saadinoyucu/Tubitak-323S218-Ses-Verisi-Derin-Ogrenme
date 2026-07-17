"""
report_utils.py
---------------
Tüm modeller ve senaryolar için özet CSV oluşturma ve heatmap grafikleri.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Tüm JSON metrik dosyalarını topla → DataFrame
# ---------------------------------------------------------------------------
def collect_all_results(results_root: str) -> pd.DataFrame:
    """
    results/
      CNN_LSTM/
        train_ASVspoof2019/
          test_ASVspoof2019_metrics.json
          ...
    yapısından tüm metrik JSON'larını okur.
    """
    rows = []
    for model_name in os.listdir(results_root):
        model_dir = os.path.join(results_root, model_name)
        if not os.path.isdir(model_dir):
            continue
        for train_scenario in os.listdir(model_dir):
            scenario_dir = os.path.join(model_dir, train_scenario)
            if not os.path.isdir(scenario_dir):
                continue
            for fname in os.listdir(scenario_dir):
                if not fname.endswith("_metrics.json"):
                    continue
                fpath = os.path.join(scenario_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    m = json.load(f)
                m["model"]          = model_name
                m["train_scenario"] = train_scenario.replace("train_", "")
                rows.append(m)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Özet CSV kaydet
# ---------------------------------------------------------------------------
def save_summary_csv(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  Özet CSV kaydedildi: {save_path}")


# ---------------------------------------------------------------------------
# Yardımcı: heatmap çiz
# ---------------------------------------------------------------------------
def _draw_heatmap(pivot: pd.DataFrame, title: str, save_path: str,
                  fmt: str = ".2f", vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.8),
                                    max(4, pivot.shape[0] * 1.2)))
    sns.heatmap(pivot.astype(float), annot=True, fmt=fmt,
                cmap="YlOrRd", vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": title, "shrink": 0.8})
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Test Set", fontsize=11)
    ax.set_ylabel("Model | Train Set", fontsize=11)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {save_path}")


# ---------------------------------------------------------------------------
# Heatmap'ler: Accuracy, F1, ROC-AUC
# ---------------------------------------------------------------------------
def generate_heatmaps(df: pd.DataFrame, summary_dir: str):
    """
    Satır: Model + Train Seti kombinasyonu
    Sütun: Test Seti
    """
    if df.empty:
        print("  [UYARI] Sonuç verisi boş, heatmap oluşturulamıyor.")
        return

    df = df.copy()
    df["row_label"] = df["model"] + " | " + df["train_scenario"]

    metrics_to_plot = {
        "accuracy":  "Accuracy Heatmap",
        "f1":        "F1-Score Heatmap",
        "precision": "Precision Heatmap",
        "recall":    "Recall Heatmap",
        "roc_auc":   "ROC-AUC Heatmap",
    }

    for metric, title in metrics_to_plot.items():
        if metric not in df.columns:
            continue
        try:
            pivot = df.pivot_table(
                index="row_label",
                columns="test_dataset",
                values=metric,
                aggfunc="mean"
            )
            save_path = os.path.join(summary_dir, f"heatmap_{metric}.png")
            _draw_heatmap(pivot, title, save_path)
        except Exception as e:
            print(f"  [UYARI] {metric} heatmap oluşturulamadı: {e}")


# ---------------------------------------------------------------------------
# Model bazlı bar karşılaştırma grafiği
# ---------------------------------------------------------------------------
def generate_model_comparison_bar(df: pd.DataFrame, summary_dir: str):
    """
    Her model için ortalama (tüm senaryo+test) metriklerini bar grafik olarak gösterir.
    """
    if df.empty:
        return

    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return

    agg = df.groupby("model")[available].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agg))
    width = 0.15
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]

    for i, metric in enumerate(available):
        bars = ax.bar(x + i * width, agg[metric], width,
                      label=metric.upper(), color=colors[i % len(colors)],
                      alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(agg["model"], fontsize=11)
    ax.set_ylim([0, 1.18])   # etiketler için yeterli boşluk
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Bazlı Ortalama Metrik Karşılaştırması",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9,
              bbox_to_anchor=(1.0, 1.0), frameon=True, borderaxespad=0)
    ax.grid(axis="y", alpha=0.3)

    # Bar etiketleri: yatay, küçük font
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8, rotation=0)

    plt.tight_layout()
    save_path = os.path.join(summary_dir, "model_comparison_bar.png")
    os.makedirs(summary_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved: {save_path}")


# ---------------------------------------------------------------------------
# Üst düzey: tüm raporları oluştur
# ---------------------------------------------------------------------------
def generate_full_report(results_root: str):
    summary_dir = os.path.join(results_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    df = collect_all_results(results_root)
    if df.empty:
        print("  [UYARI] Hiç sonuç dosyası bulunamadı.")
        return

    save_summary_csv(df, os.path.join(summary_dir, "all_models_comparison.csv"))
    generate_heatmaps(df, summary_dir)
    generate_model_comparison_bar(df, summary_dir)

    print(f"\n  ✅ Tüm özet raporlar oluşturuldu → {summary_dir}")