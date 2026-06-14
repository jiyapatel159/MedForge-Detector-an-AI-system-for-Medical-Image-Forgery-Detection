# ─────────────────────────────────────────────────────────────
# FILE: src/models/cnn_baseline.py
# PURPOSE: CNN built from scratch for medical image forgery detection
# No pretrained weights — model learns entirely from our dataset
# ─────────────────────────────────────────────────────────────

import torch                          # Main PyTorch library
import torch.nn as nn                 # Neural network building blocks


class ConvBlock(nn.Module):
    """
    One reusable building block of our CNN.
    Every block does 3 things in order:
      1. Conv2D     → detects patterns (edges, textures, shapes)
      2. BatchNorm  → stabilizes training, helps model learn faster
      3. ReLU       → activation: turns negative values to 0
                      (adds non-linearity so model can learn complex patterns)
    Think of each block as one "layer of understanding"
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()

        # Conv2D: scans the image with small filters to detect features
        # in_channels  = how many feature maps coming IN  (e.g. 3 for RGB)
        # out_channels = how many feature maps going OUT  (e.g. 32 filters)
        # kernel_size  = filter size (3×3 patch scans over image)
        # padding=1    = adds 1 pixel border so output stays same size as input
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False          # bias=False because BatchNorm handles bias
        )

        # BatchNorm: normalizes output of conv layer
        # prevents values from getting too large or too small
        self.bn = nn.BatchNorm2d(out_channels)

        # ReLU: activation function
        # f(x) = max(0, x) — keeps positive values, zeros out negatives
        # inplace=True saves memory by modifying tensor directly
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x flows: Conv → BatchNorm → ReLU
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ─────────────────────────────────────────────────────────────
class MedicalForgeryDetectorCNN(nn.Module):
    """
    Full CNN built from scratch.

    Architecture overview:
    Input: (batch, 3, 224, 224)  ← RGB image, 224×224 pixels
       ↓
    Block 1: 3   → 32  filters   ← detect basic edges
    MaxPool: 224 → 112           ← shrink spatial size by half
       ↓
    Block 2: 32  → 64  filters   ← detect simple textures
    MaxPool: 112 → 56
       ↓
    Block 3: 64  → 128 filters   ← detect complex patterns
    MaxPool: 56  → 28
       ↓
    Block 4: 128 → 256 filters   ← detect high-level features
    MaxPool: 28  → 14
       ↓
    Block 5: 256 → 512 filters   ← detect forgery-specific features
    MaxPool: 14  → 7
       ↓
    GlobalAvgPool: 7×7 → 1×1    ← compress each feature map to 1 number
       ↓
    FC(512 → 256) + Dropout(0.5) ← fully connected, prevents overfitting
    FC(256 → 128) + Dropout(0.3)
    FC(128 → 1)                  ← final output: 1 number
       ↓
    Sigmoid                      ← squash to 0-1 probability
    Output: probability of being FORGED (>0.5 = forged, <0.5 = real)
    """

    def __init__(self, dropout_rate=0.5):
        super(MedicalForgeryDetectorCNN, self).__init__()

        # ── Feature Extractor (5 convolutional blocks) ──────────────
        # Each block: Conv → BN → ReLU
        # Channels double each time: 32 → 64 → 128 → 256 → 512
        # This is standard CNN design — more filters = more complex features

        self.block1 = ConvBlock(3,   32)   # Input: RGB (3 channels)
        self.block2 = ConvBlock(32,  64)
        self.block3 = ConvBlock(64,  128)
        self.block4 = ConvBlock(128, 256)
        self.block5 = ConvBlock(256, 512)

        # MaxPool: takes the maximum value in each 2×2 window
        # Effect: shrinks image size by half, keeps strongest features
        # One pool layer shared across all blocks (same operation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # GlobalAveragePooling: averages each feature map to a single value
        # Converts (batch, 512, 7, 7) → (batch, 512, 1, 1)
        # Then we squeeze to (batch, 512)
        # Why: removes dependency on exact image size, reduces overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier (fully connected layers) ─────────────────────
        # Takes the 512 features and makes final decision

        self.classifier = nn.Sequential(

            # Layer 1: 512 inputs → 256 outputs
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),         # Normalize FC layer outputs too
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # Randomly zero 50% of neurons
                                         # Forces model to not rely on
                                         # any single feature → less overfitting

            # Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),           # Less dropout in deeper layers

            # Final output layer: 128 → 1
            # Outputs a single number (logit)
            nn.Linear(128, 1)
        )

        # Sigmoid: converts any number to range [0, 1]
        # Output interpretation:
        #   0.0 - 0.5 → Real image
        #   0.5 - 1.0 → Forged image
        self.sigmoid = nn.Sigmoid()

        # ── Weight Initialization ────────────────────────────────────
        # Good initialization helps training converge faster
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize conv layers with Kaiming (He) initialization.
        This is the standard for ReLU networks.
        Poor initialization → slow training or no learning at all.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal: sets weights based on number of inputs
                # mode='fan_out' is recommended for layers followed by ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: start with weight=1, bias=0 (identity)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # FC layers: small random weights, zero bias
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass: how data flows through the network.
        x shape at each stage shown in comments.
        """
        # Input: (batch, 3, 224, 224)

        x = self.pool(self.block1(x))   # → (batch, 32,  112, 112)
        x = self.pool(self.block2(x))   # → (batch, 64,   56,  56)
        x = self.pool(self.block3(x))   # → (batch, 128,  28,  28)
        x = self.pool(self.block4(x))   # → (batch, 256,  14,  14)
        x = self.pool(self.block5(x))   # → (batch, 512,   7,   7)

        x = self.global_avg_pool(x)     # → (batch, 512,   1,   1)
        x = x.squeeze(-1).squeeze(-1)   # → (batch, 512)
                                        # squeeze removes dimensions of size 1

        x = self.classifier(x)          # → (batch, 1)
        x = self.sigmoid(x)             # → (batch, 1) values in [0, 1]

        return x


# ─────────────────────────────────────────────────────────────
# Quick test: run this file directly to verify model works
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Testing CNN model...\n")

    # Create model instance
    model = MedicalForgeryDetectorCNN(dropout_rate=0.5)

    # Print full architecture summary
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model)

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable:,}")

    # Simulate one forward pass with fake data
    # batch_size=4, channels=3, height=224, width=224
    dummy_input = torch.randn(4, 3, 224, 224)

    model.eval()  # Set to eval mode (disables dropout for test)
    with torch.no_grad():  # Don't compute gradients during test
        output = model(dummy_input)

    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output values: {output.squeeze().tolist()}")
    print(f"\n✅ Model working correctly!")
    print(f"   Values between 0–1: {output.min().item():.4f} to "
          f"{output.max().item():.4f}")