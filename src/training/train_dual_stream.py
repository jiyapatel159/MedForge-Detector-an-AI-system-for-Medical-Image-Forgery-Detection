# ══════════════════════════════════════════════════════════════
# FILE: src/training/train_dual_stream.py
# PURPOSE: Train the Dual-Stream model (RGB + ELA)
#          Key difference from baseline:
#          - Model receives TWO inputs per sample (image + ELA)
#          - Everything else is identical to train.py
# ══════════════════════════════════════════════════════════════

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.models.dual_stream   import DualStreamForgeryDetector
from src.data.dataset_loader  import get_dataloaders

# ══════════════════════════════════════════════════════════════
CONFIG = {
    "epochs"        : 10,
    "batch_size"    : 32,
    "learning_rate" : 0.0005,   # Lower than baseline — more stable
    "dropout_rate"  : 0.5,
    "num_workers"   : 0,
    "save_dir"      : "models_saved",
    "plots_dir"     : "results/plots",
    "metrics_dir"   : "results/metrics",
}


# ══════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  ⚠️  CPU mode")
    return device


# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion,
                    optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader,
                desc=f"  Epoch {epoch} [Train]",
                leave=False, ncols=80)

    for images, elas, labels in pbar:

        # ── KEY DIFFERENCE from baseline ──────────────────────
        # Dual-stream receives TWO inputs: image AND ela
        images = images.to(device)
        elas   = elas.to(device)      # ← ELA map used here
        labels = labels.float().to(device)

        optimizer.zero_grad()

        # Forward pass with BOTH inputs
        outputs = model(images, elas).squeeze(1)

        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predicted   = (outputs > 0.5).long()
        correct    += (predicted == labels.long()).sum().item()
        total      += labels.size(0)

        if total % (10 * CONFIG["batch_size"]) == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc" : f"{correct/total*100:.1f}%"
            })

    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════
def validate(model, loader, criterion, device, split="Val"):
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, elas, labels in tqdm(loader,
                                         desc=f"  [{split}]",
                                         leave=False,
                                         ncols=80):
            images  = images.to(device)
            elas    = elas.to(device)     # ← ELA used in val too
            labels  = labels.float().to(device)

            outputs = model(images, elas).squeeze(1)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predicted   = (outputs > 0.5).long()
            correct    += (predicted == labels.long()).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════
def plot_comparison(history, baseline_csv, save_path):
    """
    Plots dual-stream curves AND overlays baseline for comparison.
    This shows clearly whether ELA stream helped.
    """
    import csv

    epochs = range(1, len(history["train_loss"]) + 1)

    # Load baseline metrics for comparison
    base_val_acc = []
    try:
        with open(baseline_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                base_val_acc.append(float(row["val_acc"]) * 100)
    except FileNotFoundError:
        base_val_acc = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dual-Stream CNN (RGB + ELA) vs Baseline",
                 fontsize=13)

    # Loss
    ax1.plot(epochs, history["train_loss"],
             "b-o", lw=2, ms=5, label="Train loss")
    ax1.plot(epochs, history["val_loss"],
             "r-o", lw=2, ms=5, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss per Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy — with baseline overlay
    dual_val = [a * 100 for a in history["val_acc"]]
    dual_tr  = [a * 100 for a in history["train_acc"]]

    ax2.plot(epochs, dual_tr,
             "b-o", lw=2, ms=5, label="Train acc (dual)")
    ax2.plot(epochs, dual_val,
             "r-o", lw=2, ms=5, label="Val acc (dual)")

    if base_val_acc:
        base_epochs = range(1, len(base_val_acc) + 1)
        ax2.plot(base_epochs, base_val_acc,
                 "g--s", lw=1.5, ms=5, alpha=0.7,
                 label="Val acc (baseline)")

    best_acc = max(dual_val)
    best_ep  = dual_val.index(best_acc) + 1
    ax2.annotate(f"Best: {best_acc:.1f}%",
                 xy=(best_ep, best_acc),
                 xytext=(best_ep + 0.3, best_acc - 5),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 color="green", fontsize=10)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy — Dual vs Baseline")
    ax2.set_ylim([40, 102])
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Comparison plot → {save_path}")


# ══════════════════════════════════════════════════════════════
def train():

    for d in [CONFIG["save_dir"],
              CONFIG["plots_dir"],
              CONFIG["metrics_dir"]]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("DUAL-STREAM TRAINING (RGB + ELA)")
    print("=" * 60)

    device = get_device()

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"]
    )
    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Val  : {len(val_loader.dataset):,} samples")

    # Model
    print("\nBuilding Dual-Stream model...")
    model  = DualStreamForgeryDetector(
                dropout_rate=CONFIG["dropout_rate"])
    model  = model.to(device)
    params = sum(p.numel() for p in model.parameters()
                 if p.requires_grad)
    print(f"  Trainable parameters: {params:,}")

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr           = CONFIG["learning_rate"],
        weight_decay = 1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=3, factor=0.5, min_lr=1e-6
    )

    # Training loop
    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    print("=" * 60)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc" : [], "val_acc" : []
    }

    best_val_acc = 0.0
    best_epoch   = 0
    start        = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):

        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, epoch
        )
        vl_loss, vl_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:02d}/{CONFIG['epochs']}  "
              f"[{time.time()-t0:.0f}s]  LR: {lr:.6f}")
        print(f"  Train → Loss: {tr_loss:.4f}  "
              f"Acc: {tr_acc*100:.2f}%")
        print(f"  Val   → Loss: {vl_loss:.4f}  "
              f"Acc: {vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_acc"    : vl_acc,
                "val_loss"   : vl_loss,
                "config"     : CONFIG,
            }, os.path.join(CONFIG["save_dir"],
                            "dual_stream_best.pth"))
            print(f"  ⭐ Best saved → val acc: {vl_acc*100:.2f}%")

    total_min = (time.time() - start) / 60
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Time              : {total_min:.1f} min")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%"
          f" (epoch {best_epoch})")

    # Plot comparison vs baseline
    plot_comparison(
        history,
        baseline_csv=f"{CONFIG['metrics_dir']}/cnn_baseline_history.csv",
        save_path   =f"{CONFIG['plots_dir']}/dual_stream_vs_baseline.png"
    )

    # Save metrics
    csv_path = f"{CONFIG['metrics_dir']}/dual_stream_history.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
        for i in range(len(history["train_loss"])):
            f.write(f"{i+1},"
                    f"{history['train_loss'][i]:.5f},"
                    f"{history['val_loss'][i]:.5f},"
                    f"{history['train_acc'][i]:.5f},"
                    f"{history['val_acc'][i]:.5f}\n")
    print(f"  📄 Metrics → {csv_path}")

    # Final test evaluation
    print("\nRunning final TEST evaluation...")
    test_loss, test_acc = validate(
        model, test_loader, criterion, device, split="Test"
    )
    print(f"\n{'='*60}")
    print(f"  BASELINE  test acc : 93.04%")
    print(f"  DUAL-STREAM test acc: {test_acc*100:.2f}%")
    diff = test_acc * 100 - 93.04
    sign = "+" if diff >= 0 else ""
    print(f"  Improvement        : {sign}{diff:.2f}%")
    print(f"{'='*60}")

    return model, history


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train()