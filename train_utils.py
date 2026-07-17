"""
train_utils.py
--------------
200 epoch eğitim döngüsü, class_weight (pos_weight) destekli BCELoss,
history CSV kaydetme, model ağırlıklarını diske yazma ve
RTX 4090 için Mixed Precision (AMP) hızlandırması.
"""

import os, csv, time
import torch
import torch.nn as nn

EPOCHS = 100

# ---------------------------------------------------------------------------
def get_criterion(pos_weight: torch.Tensor, device: torch.device) -> nn.Module:
    return WeightedBCELoss(pos_weight.to(device))

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        pred = pred.clamp(eps, 1 - eps)
        loss = -(self.pos_weight * target * torch.log(pred)
                 + (1 - target) * torch.log(1 - pred))
        return loss.mean()

# ---------------------------------------------------------------------------
def _run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)

                # ── NaN Güvenlik Duvarı ──────────────────────────
                # Loss NaN/Inf ise bu batch'i tamamen atla.
                # Optimizer adımı yapılmaz → model ağırlıkları bozulmaz.
                if not torch.isfinite(loss):
                    nan_batches += 1
                    continue

                loss.backward()
                # Gradyanların patlamasını (NaN loss) engelle
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                if not torch.isfinite(loss):
                    nan_batches += 1
                    continue

            total_loss += loss.item() * len(y_batch)
            predicted = (preds >= 0.5).float()
            correct   += (predicted == y_batch).sum().item()
            total     += len(y_batch)

    if total == 0:
        return float("nan"), 0.0
    avg_loss = total_loss / total
    acc      = correct / total
    return avg_loss, acc

# ---------------------------------------------------------------------------
def train_model(model: nn.Module,
                train_loader,
                pos_weight: torch.Tensor,
                device: torch.device,
                save_dir: str) -> list[dict]:
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    criterion = get_criterion(pos_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_loss = float("inf")
    best_state = None

    print(f"\n{'='*60}")
    print(f"  Eğitim başlıyor → {EPOCHS} epoch | Device: {device} | AMP: Kapalı (Sabitlik için)")
    print(f"{'='*60}")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion,
                                     optimizer, device, train=True)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc,  6),
            "lr":         round(scheduler.get_last_lr()[0], 8),
        })

        if tr_loss < best_loss and not torch.isnan(torch.tensor(tr_loss)):
            best_loss  = tr_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            print(f"  Epoch [{epoch:>3}/{EPOCHS}]  "
                  f"Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  "
                  f"({elapsed:.1f} dk)")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, os.path.join(save_dir, "model_weights.pt"))
    _save_history_csv(history, os.path.join(save_dir, "train_history.csv"))

    total_time = (time.time() - t0) / 60
    print(f"\n  Eğitim tamamlandı ({total_time:.1f} dk). "
          f"En iyi train loss: {best_loss:.4f}")
    print(f"  Ağırlıklar kaydedildi: {save_dir}/model_weights.pt\n")

    return history

def _save_history_csv(history: list[dict], path: str):
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
