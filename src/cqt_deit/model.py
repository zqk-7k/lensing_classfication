"""CQT--DeiT baseline model."""

import timm
import torch.nn as nn


class FeatureGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return features * self.fc(features)


def build_cqt_deit(num_classes=2, pretrained=True, hidden_dim=512,
                   dropout_rate=0.5, freeze_backbone=False):
    """Build the SEMD-inspired CQT--DeiT classifier used in the comparison."""
    model = timm.create_model(
        "deit_tiny_distilled_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    embedding_dim = model.embed_dim
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    def classification_head():
        return nn.Sequential(
            nn.LayerNorm(embedding_dim),
            FeatureGate(embedding_dim, reduction=4),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    model.head = classification_head()
    model.head_dist = classification_head()
    return model
