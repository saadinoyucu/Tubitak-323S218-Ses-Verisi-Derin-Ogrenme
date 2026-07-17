"""
plot_utils.py
-------------
Eğitim geçmişinden (history) grafik oluşturma ve kaydetme.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Train Loss & Accuracy Grafiği
# ---------------------------------------------------------------------------
def plot_train_history(history: list[dict], save_dir: str):
    """
    history: [{"epoch": 1, "train_loss": ..., "train_acc": ...}, ...]
    Her model/train kombinasyonu için:
      - train_loss_acc.png → Loss ve Accuracy eğrileri yan yana
    """
    if not history:
        return

    df = pd.DataFrame(history)
    epochs = df["epoch"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color="#E53935", lw=1.8,
            label="Train Loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Eğitim Loss Eğrisi", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(epochs, df["train_acc"], color="#1E88E5", lw=1.8,
            label="Train Acc")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.set_title("Eğitim Accuracy Eğrisi", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "train_loss_acc.png")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik kaydedildi: {save_path}")
