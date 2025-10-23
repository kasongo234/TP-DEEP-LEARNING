# models.py
import torch
import torch.nn as nn
from torchvision import models

class CNNFromScratch(nn.Module):
    """
    Version légère (CPU-friendly), toujours 3 blocs Conv+BN+ReLU+MaxPool.
    Canaux: 16 -> 32 -> 64, + AdaptiveAvgPool2d pour indépendance à la taille d'entrée.
    Dropout dans la tête fully-connected.
    """
    def __init__(self, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),         # -> 64
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_transfer_model(
    num_classes=2,
    backbone="resnet18",
    pretrained=True,
    freeze_backbone=True,
    dropout_p=0.5,
):
    """Transfert learning : ResNet18 par défaut (tête remplacée)."""
    b = backbone.lower()
    if b == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )
        return model
    raise ValueError(f"Backbone non supporté: {backbone}")
