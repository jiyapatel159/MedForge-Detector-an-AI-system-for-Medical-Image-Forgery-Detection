# ─────────────────────────────────────────────────────────────
# FILE: src/training/train.py
# PURPOSE: Full training loop for the CNN forgery detector
# ─────────────────────────────────────────────────────────────

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')                  # Non-interactive backend (no popup)
import matplotlib.pyplot as plt
from tqdm import tqdm                  # Progress bar

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.models.cnn_baseline import MedicalForgeryDetectorCNN
from src.data.dataset_loader import get_dataloaders


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — change these values to experiment
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "epochs"       : 10,          # Number of times to go through full dataset
    "batch_size"   : 32,          # Images per training step
    "learning_rate": 0.001,       # How fast model updates weights (Adam default)
    "dropout_rate" : 0.5,         # Fraction of neurons to drop during training
    "num_workers"  : 0,           # 0 = no multiprocessing (safe for Windows)
    "save_dir"     : "models_saved",
    "results_dir"  : "results",
}

# ─────────────────────────────────────────────────────────────
def get_device():
    """
    Automatically use GPU if available, else CPU.
    GPU training is ~10-50x faster than CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not found — using CPU (training will be slower)")
    return device


# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs ONE complete pass through the training dataset.

    Steps for each batch:
      1. Load batch of images + labels
      2. Forward pass: model makes predictions
      3. Compute loss: how wrong were the predictions?
      4. Backward pass: compute gradients (how to fix weights)
      5. Optimizer step: update weights using gradients
      6. Zero gradients: reset for next batch
    """
    model.train()   # Enable training mode (activates Dropout, BatchNorm)

    running_loss    = 0.0   # Accumulate loss across all batches
    correct         = 0     # Count correct predictions
    total           = 0     # Count total samples

    # tqdm wraps the loader to show a progress bar
    for images, elas, labels in tqdm(loader, desc="  Training",
                                     leave=False):

        # Move data to same device as model (GPU or CPU)
        # images: (batch, 3, 224, 224)
        # labels: (batch,) — 0=real, 1=forged
        images = images.to(device)
        labels = labels.float().to(device)  # float needed for BCELoss

        # ── Forward Pass ──────────────────────────────────────
        # Zero gradients BEFORE forward pass
        # (PyTorch accumulates gradients — must clear each step)
        optimizer.zero_grad()

        # Run images through model → get predictions
        outputs = model(images)           # shape: (batch, 1)
        outputs = outputs.squeeze(1)      # shape: (batch,) — remove extra dim

        # ── Compute Loss ──────────────────────────────────────
        # BCELoss compares predictions (0-1 float) with labels (0 or 1)
        # High loss = model is very wrong
        # Low loss  = model is mostly right
        loss = criterion(outputs, labels)

        # ── Backward Pass ─────────────────────────────────────
        # Compute gradients: how should each weight change to reduce loss?
        loss.backward()

        # Gradient clipping: prevents exploding gradients
        # Caps gradient magnitude at 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── Update Weights ────────────────────────────────────
        # Adam optimizer uses gradients to update all weights
        optimizer.step()

        # ── Track metrics ─────────────────────────────────────
        running_loss += loss.item() * images.size(0)  # Total loss this batch

        # Convert probabilities to binary predictions
        # threshold = 0.5: above → forged (1), below → real (0)
        predicted = (outputs > 0.5).long()
        correct  += (predicted == labels.long()).sum().item()
        total    += labels.size(0)

    # Average loss and accuracy over entire epoch
    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ─────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    """
    Evaluate model on validation set.
    NO weight updates — we only measure performance.
    """
    model.eval()   # Disable Dropout, use running stats for BatchNorm

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():  # Don't compute gradients (saves memory + speed)
        for images, elas, labels in tqdm(loader, desc="  Validating",
                                         leave=False):
            images = images.to(device)
            labels = labels.float().to(device)

            outputs  = model(images).squeeze(1)
            loss     = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted     = (outputs > 0.5).long()
            correct      += (predicted == labels.long()).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
def plot_curves(history, save_path):
    """
    Plot training and validation accuracy + loss curves.
    Saves as PNG — tells us if model is learning or overfitting.

    Overfitting = train accuracy keeps rising but val accuracy stops
    Good fit    = both curves rise together and level off
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Baseline — Training Results", fontsize=14, y=1.02)

    # ── Loss Plot ─────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"],
                 'b-o', linewidth=2, markersize=5, label="Train loss")
    axes[0].plot(epochs, history["val_loss"],
                 'r-o', linewidth=2, markersize=5, label="Val loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy Plot ─────────────────────────────────────────
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]],
                 'b-o', linewidth=2, markersize=5, label="Train acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],
                 'r-o', linewidth=2, markersize=5, label="Val acc")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim([40, 100])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Curves saved: {save_path}")


# ─────────────────────────────────────────────────────────────
def train():
    """Main training function — runs everything"""

    # Setup
    os.makedirs(CONFIG["save_dir"],   exist_ok=True)
    os.makedirs(CONFIG["results_dir"] + "/plots",   exist_ok=True)
    os.makedirs(CONFIG["results_dir"] + "/metrics", exist_ok=True)

    device = get_device()

    # ── Data ──────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"]
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    print("\nBuilding model...")
    model = MedicalForgeryDetectorCNN(dropout_rate=CONFIG["dropout_rate"])
    model = model.to(device)   # Move model to GPU/CPU

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {params:,}")

    # ── Loss Function ─────────────────────────────────────────
    # BCELoss: Binary Cross Entropy
    # Used for binary classification (2 classes)
    # Measures difference between predicted probability and true label
    criterion = nn.BCELoss()

    # ── Optimizer ─────────────────────────────────────────────
    # Adam: adaptive learning rate optimizer
    # Combines momentum + RMSProp
    # lr=0.001 is the standard starting point for Adam
    optimizer = optim.Adam(
        model.parameters(),
        lr           = CONFIG["learning_rate"],
        weight_decay = 1e-4     # L2 regularization: penalizes large weights
    )

    # ── Learning Rate Scheduler ───────────────────────────────
    # Reduces learning rate when validation loss stops improving
    # patience=3 = wait 3 epochs before reducing LR
    # factor=0.5 = multiply LR by 0.5 (halve it)
    # min_lr = never go below this
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = 'min',    # Monitor loss (lower = better)
        patience  = 3,
        factor    = 0.5,
        min_lr    = 1e-6,
        verbose   = True
    )

    # ── Training Loop ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc" : [], "val_acc" : []
    }

    best_val_acc  = 0.0
    best_val_loss = float('inf')

    for epoch in range(1, CONFIG["epochs"] + 1):

        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 40)

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate scheduler
        # (pass val_loss so it knows if we're improving)
        scheduler.step(val_loss)

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print epoch results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"  Val   → Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}%")
        print(f"  LR    : {current_lr:.6f}")

        # Save best model (based on val accuracy)
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            save_path = os.path.join(CONFIG["save_dir"],
                                     "cnn_baseline_best.pth")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_acc"    : val_acc,
                "val_loss"   : val_loss,
                "config"     : CONFIG,
            }, save_path)
            print(f"  ⭐ New best model saved! Val acc: {val_acc*100:.2f}%")

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Best val loss     : {best_val_loss:.4f}")
    print(f"  Model saved to    : {CONFIG['save_dir']}/cnn_baseline_best.pth")

    # Plot and save curves
    plot_curves(
        history,
        save_path=f"{CONFIG['results_dir']}/plots/cnn_baseline_curves.png"
    )

    # Save raw numbers to file
    metrics_path = f"{CONFIG['results_dir']}/metrics/cnn_baseline_history.txt"
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
        for i in range(len(history["train_loss"])):
            f.write(f"{i+1},"
                    f"{history['train_loss'][i]:.4f},"
                    f"{history['val_loss'][i]:.4f},"
                    f"{history['train_acc'][i]:.4f},"
                    f"{history['val_acc'][i]:.4f}\n")
    print(f"  📄 Metrics saved to: {metrics_path}")

    return model, history


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()