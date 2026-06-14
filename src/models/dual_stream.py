import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class StreamA_RGB(nn.Module):
    def __init__(self):
        super(StreamA_RGB, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32,  64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64,  128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class StreamB_ELA(nn.Module):
    def __init__(self):
        super(StreamB_ELA, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(3,  32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class DualStreamForgeryDetector(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DualStreamForgeryDetector, self).__init__()

        self.stream_a = StreamA_RGB()
        self.stream_b = StreamB_ELA()

        fusion_size = 512 + 256

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, ela):
        feat_rgb = self.stream_a(image)
        feat_ela = self.stream_b(ela)
        fused    = torch.cat([feat_rgb, feat_ela], dim=1)
        out      = self.classifier(fused)
        out      = self.sigmoid(out)
        return out


if __name__ == "__main__":
    print("Testing Dual-Stream model...")
    model     = DualStreamForgeryDetector(dropout_rate=0.5)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    dummy_img = torch.randn(4, 3, 224, 224)
    dummy_ela = torch.randn(4, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_ela)

    print(f"Image input shape : {dummy_img.shape}")
    print(f"ELA   input shape : {dummy_ela.shape}")
    print(f"Output shape      : {output.shape}")
    print(f"Output values     : {output.squeeze().tolist()}")
    print(f"\n✅ Dual-Stream model working correctly!")