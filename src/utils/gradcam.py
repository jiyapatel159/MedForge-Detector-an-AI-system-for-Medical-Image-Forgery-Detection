# ══════════════════════════════════════════════════════════════
# FILE: src/utils/gradcam.py
# PURPOSE: Grad-CAM visualization for CNN Baseline model only
# ══════════════════════════════════════════════════════════════

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.models.cnn_baseline import MedicalForgeryDetectorCNN


# ══════════════════════════════════════════════════════════════
class GradCAM:
    """
    Grad-CAM for CNN Baseline model.
    Hooks into the last Conv2d layer of block5.
    Red regions = where model detected forgery.
    """

    def __init__(self, model):
        self.model       = model
        self.activations = {}
        self.gradients   = {}

        # Auto-detect correct target layer
        self.target = self._find_target_layer()
        print(f"  Grad-CAM target: {self.target}")

        self._register_hooks()

    def _find_target_layer(self):
        """Find last Conv2d — works regardless of attribute name"""
        all_convs = [
            m for m in self.model.modules()
            if isinstance(m, torch.nn.Conv2d)
        ]
        if not all_convs:
            raise ValueError("No Conv2d layer found in model")
        # Last Conv2d = deepest, most semantic features
        return all_convs[-1]

    def _register_hooks(self):
        def fwd_hook(m, i, o):
            self.activations["feat"] = o.clone().detach()

        def bwd_hook(m, gi, go):
            self.gradients["grad"] = go[0].clone().detach()

        self.target.register_forward_hook(fwd_hook)
        self.target.register_full_backward_hook(bwd_hook)

    def generate(self, image_tensor):
        """
        Generate heatmap for one image.
        Returns: heatmap (224x224), prediction probability, is_forged bool
        """
        self.model.eval()
        self.activations.clear()
        self.gradients.clear()

        img_in = image_tensor.unsqueeze(0).clone().detach()
        img_in.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(img_in)
        pred   = output.item()

        output.backward()

        if "grad" not in self.gradients or \
           "feat" not in self.activations:
            raise ValueError("Hooks did not capture data")

        grads   = self.gradients["grad"][0]     # (C, H, W)
        acts    = self.activations["feat"][0]   # (C, H, W)
        weights = grads.mean(dim=(1, 2))        # (C,)

        cam = torch.zeros(acts.shape[1:])
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224))
        return cam, pred, pred > 0.5


# ══════════════════════════════════════════════════════════════
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def overlay_heatmap(original_pil, heatmap, alpha=0.45):
    """Blend Grad-CAM heatmap onto original image"""
    orig_np       = np.array(
        original_pil.convert('RGB').resize((224, 224))
    )
    heatmap_u8    = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8,
                                      cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color,
                                  cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        orig_np, 1 - alpha,
        heatmap_color, alpha, 0
    )
    return Image.fromarray(overlay)


# ══════════════════════════════════════════════════════════════
def visualize_gradcam_batch(model_path, num_samples=6):
    """
    Generate Grad-CAM for sample test images.
    Shows: Original | ELA Map | Grad-CAM Overlay
    Saves to results/gradcam_outputs/
    """
    os.makedirs("results/gradcam_outputs", exist_ok=True)

    device    = torch.device("cpu")
    transform = get_transform()

    # Load CNN baseline model
    print("Loading CNN Baseline model...")
    model = MedicalForgeryDetectorCNN()
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    gradcam = GradCAM(model)

    # Collect test samples
    test_real   = "data/processed/test/real"
    test_forged = "data/processed/test/forged"

    samples = []

    real_files = [
        f for f in os.listdir(test_real)
        if f.endswith('.png')
    ][:3]
    for fname in real_files:
        samples.append({
            "img_path"  : os.path.join(test_real, fname),
            "true_label": "Real"
        })

    forged_files = [
        f for f in os.listdir(test_forged)
        if f.endswith('.png')
    ][:3]
    for fname in forged_files:
        samples.append({
            "img_path"  : os.path.join(test_forged, fname),
            "true_label": "Forged"
        })

    samples = samples[:num_samples]
    print(f"Generating Grad-CAM for {len(samples)} samples...")

    # Plot grid: rows = samples, cols = Original | Grad-CAM
    fig, axes = plt.subplots(
        len(samples), 2,
        figsize=(8, 4 * len(samples))
    )
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    axes[0][0].set_title("Original Image",
                          fontsize=12, fontweight='bold')
    axes[0][1].set_title("Grad-CAM Overlay",
                          fontsize=12, fontweight='bold')

    for idx, sample in enumerate(samples):
        orig_pil   = Image.open(
            sample["img_path"]
        ).convert('RGB')
        orig_np    = np.array(orig_pil.resize((224, 224)))
        img_tensor = transform(orig_pil)

        heatmap, prob, is_forged = gradcam.generate(img_tensor)
        overlay = overlay_heatmap(orig_pil, heatmap)

        pred_label = "FORGED" if is_forged else "REAL"
        true_label = sample["true_label"].upper()
        correct    = "✓" if pred_label == true_label else "✗"
        color      = "green" if pred_label == true_label \
                     else "red"

        row_label  = (
            f"True: {true_label}\n"
            f"Pred: {pred_label} "
            f"({prob*100:.1f}%) {correct}"
        )

        axes[idx][0].imshow(orig_np)
        axes[idx][0].set_ylabel(row_label,
                                 fontsize=9, color=color)
        axes[idx][0].axis('off')

        axes[idx][1].imshow(overlay)
        axes[idx][1].axis('off')

        print(
            f"  [{idx+1}] True: {true_label:<8} "
            f"Pred: {pred_label:<8} "
            f"({prob*100:.1f}%) {correct}"
        )

    plt.suptitle(
        "Grad-CAM — CNN Baseline\n"
        "Red = where model detected forgery",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    save_path = "results/gradcam_outputs/gradcam_baseline.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Saved → {save_path}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    visualize_gradcam_batch(
        model_path  = "models_saved/cnn_baseline_best.pth",
        num_samples = 6
    )