"""
eval_utils.py
-------------
Her test seti için metrik hesaplama, JSON kaydetme, ROC eğrisi ve
confusion matrix çizimi.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")                      # GUI olmayan ortamda çalışmak için
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# ---------------------------------------------------------------------------
# Model ile tahmin üret
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device):
    """
    Döndürür:
      y_true  : np.ndarray (N,)
      y_prob  : np.ndarray (N,) — tahmin olasılıkları (patient=1 sınıfı)
      y_pred  : np.ndarray (N,) — 0.5 eşiğiyle binary tahminler
    """
    model.eval()
    model.to(device)

    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        probs = model(X_batch).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.squeeze(1).cpu().numpy())

    y_prob  = np.concatenate(all_probs)
    y_true  = np.concatenate(all_labels)
    y_pred  = (y_prob >= 0.5).astype(np.float32)
    return y_true, y_prob, y_pred


# ---------------------------------------------------------------------------
# Metrik hesaplama
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, y_pred) -> dict:
    metrics = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)),  6),
        "f1":        round(float(f1_score(y_true, y_pred,
                                          zero_division=0)),         6),
        "precision": round(float(precision_score(y_true, y_pred,
                                                 zero_division=0)),  6),
        "recall":    round(float(recall_score(y_true, y_pred,
                                              zero_division=0)),     6),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
    except ValueError:
        metrics["roc_auc"] = None   # Tek sınıf varsa hesaplanamaz

    return metrics


# ---------------------------------------------------------------------------
# JSON kaydet
# ---------------------------------------------------------------------------
def save_metrics_json(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Confusion Matrix çizimi
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, test_name: str, save_path: str, feature_name: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Kontrol (0)", "Hasta (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Tahmin Edilen", fontsize=11)
    ax.set_ylabel("Gerçek", fontsize=11)
    title = f"Karışıklık Matrisi\n{test_name}"
    if feature_name:
        title += f" - Öznitelik: {feature_name}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC Eğrisi çizimi
# ---------------------------------------------------------------------------
def plot_roc_curve(y_true, y_prob, test_name: str, save_path: str,
                   roc_auc: float | None, feature_name: str | None = None):
    fig, ax = plt.subplots(figsize=(5, 4))

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_label   = f"AUC = {roc_auc:.4f}" if roc_auc else "AUC = N/A"
        ax.plot(fpr, tpr, color="#2196F3", lw=2, label=auc_label)
    except ValueError:
        ax.text(0.5, 0.5, "Tek sınıf – ROC hesaplanamadı",
                ha="center", va="center", transform=ax.transAxes)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Yanlış Pozitif Oranı", fontsize=11)
    ax.set_ylabel("Doğru Pozitif Oranı", fontsize=11)
    title = f"ROC Eğrisi\n{test_name}"
    if feature_name:
        title += f" - Öznitelik: {feature_name}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Üst düzey: bir test seti üzerinde tam değerlendirme
# ---------------------------------------------------------------------------
def evaluate_on_testset(model: nn.Module,
                        test_loader,
                        test_name: str,
                        save_dir: str,
                        device: torch.device,
                        feature_name: str | None = None) -> dict:
    """
    Tek bir test seti için:
      1. Tahmin üret
      2. Metrikleri hesapla & JSON kaydet
      3. Confusion matrix PNG kaydet
      4. ROC eğrisi PNG kaydet
    Döndürür: metrik dict
    """
    y_true, y_prob, y_pred = predict(model, test_loader, device)

    metrics = compute_metrics(y_true, y_prob, y_pred)
    metrics["test_dataset"] = test_name
    metrics["n_samples"]    = int(len(y_true))
    metrics["n_patient"]    = int(y_true.sum())
    metrics["n_control"]    = int((y_true == 0).sum())

    # Dosya adı için test_name temizle
    safe_name = test_name.replace(" ", "_").replace("/", "-")

    save_metrics_json(
        metrics,
        os.path.join(save_dir, f"test_{safe_name}_metrics.json")
    )
    plot_confusion_matrix(
        y_true, y_pred, test_name,
        os.path.join(save_dir, f"test_{safe_name}_confusion.png"),
        feature_name=feature_name,
    )
    plot_roc_curve(
        y_true, y_prob, test_name,
        os.path.join(save_dir, f"test_{safe_name}_roc.png"),
        metrics.get("roc_auc"),
        feature_name=feature_name,
    )

    # Konsol özeti
    print(f"  [{test_name}]  "
          f"Acc={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}  "
          f"Pre={metrics['precision']:.4f}  "
          f"Rec={metrics['recall']:.4f}  "
          f"AUC={metrics.get('roc_auc') or 'N/A'}")

    return metrics
