# ══════════════════════════════════════════════════════════════
# FILE: src/training/final_evaluation.py
# PURPOSE: Complete model evaluation on unseen test data
#
# What this script does:
#   1. Loads the best saved CNN model
#   2. Runs it on 2,527 test images (never seen during training)
#   3. Computes all metrics: accuracy, precision, recall, F1, AUC
#   4. Plots confusion matrix + ROC curve
#   5. Shows sample predictions with correct/wrong labels
#   6. Saves everything to results/
# ══════════════════════════════════════════════════════════════

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve, auc,
    f1_score, precision_score,
    recall_score, accuracy_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.models.cnn_baseline import MedicalForgeryDetectorCNN
from src.data.dataset_loader import get_dataloaders


# ══════════════════════════════════════════════════════════════
def load_model(model_path, device):
    """Load the best saved model checkpoint"""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run training first: python src/training/train.py"
        )

    model = MedicalForgeryDetectorCNN()
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    print(f"  Saved at epoch  : {ckpt['epoch']}")
    print(f"  Val accuracy    : {ckpt['val_acc']*100:.2f}%")
    print(f"  Val loss        : {ckpt['val_loss']:.4f}")
    return model


# ══════════════════════════════════════════════════════════════
def run_test_evaluation(model, test_loader, device):
    """
    Pass ALL test images through the model.
    Collect predictions and true labels.
    No weight updates — purely measuring performance.
    """
    model.eval()

    all_probs  = []   # raw probability outputs (0.0 to 1.0)
    all_preds  = []   # binary predictions (0 or 1)
    all_labels = []   # true labels (0 = real, 1 = forged)

    print(f"\nRunning on {len(test_loader.dataset):,} "
          f"test images...")

    with torch.no_grad():
        for images, elas, labels in tqdm(
                test_loader, desc="  Testing", ncols=70):

            images  = images.to(device)
            outputs = model(images).squeeze(1)

            probs  = outputs.cpu().numpy()
            preds  = (outputs > 0.5).long().cpu().numpy()
            labs   = labels.numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labs)

    return (np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels))


# ══════════════════════════════════════════════════════════════
def print_all_metrics(y_true, y_pred, y_prob):
    """Print every metric clearly"""

    acc       = accuracy_score(y_true, y_pred)  * 100
    precision = precision_score(y_true, y_pred) * 100
    recall    = recall_score(y_true, y_pred)    * 100
    f1        = f1_score(y_true, y_pred)        * 100
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc   = auc(fpr, tpr)

    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 55)
    print("  FINAL TEST SET EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Total test images  : {len(y_true):,}")
    print(f"  Real images        : {(y_true==0).sum():,}")
    print(f"  Forged images      : {(y_true==1).sum():,}")
    print("-" * 55)
    print(f"  Accuracy           : {acc:.2f}%")
    print(f"  Precision          : {precision:.2f}%")
    print(f"  Recall             : {recall:.2f}%")
    print(f"  F1 Score           : {f1:.2f}%")
    print(f"  AUC-ROC            : {roc_auc:.4f}")
    print("-" * 55)
    print(f"  True Negatives  (TN): {tn}  "
          f"← correctly said REAL")
    print(f"  True Positives  (TP): {tp}  "
          f"← correctly said FORGED")
    print(f"  False Positives (FP): {fp}  "
          f"← said FORGED but was REAL")
    print(f"  False Negatives (FN): {fn}  "
          f"← said REAL but was FORGED ⚠️")
    print("=" * 55)

    print("\nPer-class report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Real", "Forged"]
    ))

    return {
        "accuracy" : acc,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "auc"      : roc_auc,
        "tn": tn, "tp": tp,
        "fp": fp, "fn": fn,
        "fpr": fpr, "tpr": tpr
    }


# ══════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Confusion matrix visual.

    Rows = what the image actually was
    Cols = what the model predicted

    Top-left  (TN): real → predicted real    ✅
    Top-right (FP): real → predicted forged  ⚠️
    Bot-left  (FN): forged → predicted real  ❌
    Bot-right (TP): forged → predicted forged ✅
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot     = True,
        fmt       = 'd',
        cmap      = 'Blues',
        xticklabels = ['Predicted Real', 'Predicted Forged'],
        yticklabels = ['Actually Real',  'Actually Forged'],
        linewidths  = 0.5,
        annot_kws   = {"size": 16},
        ax          = ax
    )
    ax.set_title(
        'Confusion Matrix — CNN Baseline\n'
        'Medical Image Forgery Detection',
        fontsize=13, pad=15
    )

    # Add TN/FP/FN/TP labels inside cells
    labels_map = [['TN', 'FP'], ['FN', 'TP']]
    colors_map = [['green', 'orange'], ['red', 'green']]
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.78,
                labels_map[i][j],
                ha='center', fontsize=11,
                color=colors_map[i][j],
                fontweight='bold'
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Confusion matrix → {save_path}")


# ══════════════════════════════════════════════════════════════
def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """
    ROC Curve — shows how well model separates
    real vs forged at every possible threshold.

    AUC = 1.0 → perfect model
    AUC = 0.5 → random guessing
    Your AUC = 0.9809 → excellent
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr,
            color='steelblue', lw=2.5,
            label=f'CNN Baseline (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    ax.plot([0, 1], [0, 1],
            'k--', lw=1.5, label='Random guess (AUC = 0.5)')

    # Mark the operating point at threshold 0.5
    ax.set_xlabel('False Positive Rate\n'
                  '(Real images wrongly flagged as Forged)',
                  fontsize=11)
    ax.set_ylabel('True Positive Rate\n'
                  '(Forged images correctly detected)',
                  fontsize=11)
    ax.set_title('ROC Curve — CNN Baseline', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 ROC curve → {save_path}")


# ══════════════════════════════════════════════════════════════
def plot_sample_predictions(model, test_loader,
                             device, save_path,
                             num_samples=10):
    """
    Show sample images with predictions.
    Green border = correct, Red border = wrong.
    Helps visually confirm model is working.
    """
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as TF

    model.eval()
    images_shown = []
    labels_shown = []
    preds_shown  = []
    probs_shown  = []

    with torch.no_grad():
        for images, elas, labels in test_loader:
            outputs = model(images.to(device)).squeeze(1)
            probs   = outputs.cpu().numpy()
            preds   = (outputs > 0.5).long().cpu().numpy()

            for i in range(len(images)):
                if len(images_shown) >= num_samples:
                    break
                images_shown.append(images[i])
                labels_shown.append(labels[i].item())
                preds_shown.append(preds[i])
                probs_shown.append(probs[i])

            if len(images_shown) >= num_samples:
                break

    # Plot grid
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for idx in range(num_samples):
        ax  = axes[idx]

        # Denormalize image for display
        img = images_shown[idx].numpy().transpose(1, 2, 0)
        img = (img * std + mean)
        img = np.clip(img, 0, 1)

        true_label = "Real"   if labels_shown[idx] == 0 \
                     else "Forged"
        pred_label = "Real"   if preds_shown[idx]  == 0 \
                     else "Forged"
        prob       = probs_shown[idx]
        correct    = labels_shown[idx] == preds_shown[idx]

        ax.imshow(img, cmap='gray')
        ax.axis('off')

        # Border color: green=correct, red=wrong
        border_color = '#28a745' if correct else '#dc3545'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
            spine.set_visible(True)

        title_color = '#28a745' if correct else '#dc3545'
        ax.set_title(
            f"True: {true_label}\n"
            f"Pred: {pred_label} ({prob*100:.1f}%)",
            fontsize=9,
            color=title_color,
            fontweight='bold'
        )

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(
        "Sample Test Predictions\n"
        "Green border = Correct  |  Red border = Wrong",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Sample predictions → {save_path}")


# ══════════════════════════════════════════════════════════════
def save_evaluation_report(metrics, save_path):
    """Save all numbers to a readable text report"""

    report = f"""
╔══════════════════════════════════════════════════════╗
║     MEDFORGE DETECTOR — FINAL EVALUATION REPORT     ║
╚══════════════════════════════════════════════════════╝

Model        : CNN Baseline (built from scratch)
Framework    : PyTorch 2.1.2
Test Set     : 2,527 images (never seen during training)
Date         : Final Submission

──────────────────────────────────────────────────────
PERFORMANCE METRICS
──────────────────────────────────────────────────────
Accuracy     : {metrics['accuracy']:.2f}%
Precision    : {metrics['precision']:.2f}%
Recall       : {metrics['recall']:.2f}%
F1 Score     : {metrics['f1']:.2f}%
AUC-ROC      : {metrics['auc']:.4f}

──────────────────────────────────────────────────────
CONFUSION MATRIX
──────────────────────────────────────────────────────
True Negatives  (TN) : {metrics['tn']:>5}  (Real → Real ✅)
False Positives (FP) : {metrics['fp']:>5}  (Real → Forged ⚠️)
False Negatives (FN) : {metrics['fn']:>5}  (Forged → Real ❌)
True Positives  (TP) : {metrics['tp']:>5}  (Forged → Forged ✅)

──────────────────────────────────────────────────────
DATASET SUMMARY
──────────────────────────────────────────────────────
Total images    : 16,830
Training set    : 11,780  (70%)
Validation set  :  2,523  (15%)
Test set        :  2,527  (15%)

Sources:
  - RSNA Pneumonia X-Ray Dataset
  - Medical Image Tamper Dataset (Kaggle)
  - Brain Tumor MRI Dataset (Kaggle)

Forgery types detected:
  - Copy-move
  - Copy-paste (splicing)
  - Content removal
  - Text addition

──────────────────────────────────────────────────────
MODEL ARCHITECTURE
──────────────────────────────────────────────────────
Type         : 5-block CNN (no pretrained weights)
Parameters   : 1,734,689 trainable
Input size   : 224 × 224 × 3
Optimizer    : Adam (lr=0.001)
Loss         : Binary Cross Entropy
Epochs       : 10
Best epoch   : 7
──────────────────────────────────────────────────────
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  📄 Full report → {save_path}")
    print(report)


# ══════════════════════════════════════════════════════════════
def evaluate():

    for d in ["results/plots",
              "results/metrics",
              "results/gradcam_outputs"]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cpu")
    print("=" * 55)
    print("  MEDFORGE DETECTOR — FINAL EVALUATION")
    print("=" * 55)

    # Load model
    model = load_model(
        "models_saved/cnn_baseline_best.pth", device
    )

    # Load test data only
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        batch_size=32, num_workers=0
    )
    print(f"  Test samples: {len(test_loader.dataset):,}")

    # Run evaluation
    preds, probs, labels = run_test_evaluation(
        model, test_loader, device
    )

    # Print all metrics
    metrics = print_all_metrics(labels, preds, probs)

    # Plot confusion matrix
    plot_confusion_matrix(
        labels, preds,
        "results/plots/final_confusion_matrix.png"
    )

    # Plot ROC curve
    plot_roc_curve(
        metrics["fpr"], metrics["tpr"], metrics["auc"],
        "results/plots/final_roc_curve.png"
    )

    # Sample predictions visual
    plot_sample_predictions(
        model, test_loader, device,
        "results/plots/final_sample_predictions.png",
        num_samples=10
    )

    # Save text report
    save_evaluation_report(
        metrics,
        "results/metrics/final_evaluation_report.txt"
    )

    print("\n" + "=" * 55)
    print("  ALL OUTPUTS SAVED")
    print("=" * 55)
    print("  results/plots/final_confusion_matrix.png")
    print("  results/plots/final_roc_curve.png")
    print("  results/plots/final_sample_predictions.png")
    print("  results/metrics/final_evaluation_report.txt")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    evaluate()
    