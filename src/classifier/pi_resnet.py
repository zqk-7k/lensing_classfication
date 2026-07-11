"""One-dimensional Siamese residual network for event-pair ranking."""

import torch
import torch.nn as nn


class Snake(nn.Module):
    def __init__(self, _features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return x + torch.sin(self.alpha * x).pow(2) / (self.alpha + 1e-9)


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(x.mean(dim=2)).view(x.size(0), x.size(1), 1)
        return x * weights


class PeriodicResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=15, alpha=1.0, use_snake=True, use_se=True):
        super().__init__()
        activation = Snake(channels, alpha) if use_snake else nn.ReLU()
        attention = SEBlock1D(channels) if use_se else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels), activation,
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels), attention,
        )
        self.act = Snake(channels, alpha) if use_snake else nn.ReLU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class PIResNet(nn.Module):
    """Shared 1D encoder with lensing-motivated pairwise interactions."""

    def __init__(self, in_channels=1, d_model=256, width_scale=4.0,
                 use_snake=True, use_se=True, use_pairwise_fusion=True):
        super().__init__()
        # Keep the historical attribute name so released checkpoints remain compatible.
        self.use_physics_fusion = use_pairwise_fusion
        base = int(32 * width_scale)
        activation = lambda channels: Snake(channels) if use_snake else nn.ReLU()
        block = lambda channels, kernel: PeriodicResidualBlock(
            channels, kernel, use_snake=use_snake, use_se=use_se
        )
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, stride=2, padding=7),
            nn.BatchNorm1d(base), activation(base), nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            block(base, 15), block(base, 15), nn.Conv1d(base, base * 2, 11, stride=2, padding=5),
            nn.BatchNorm1d(base * 2), activation(base * 2), block(base * 2, 11),
        )
        self.stage2 = nn.Sequential(
            block(base * 2, 11), block(base * 2, 11),
            nn.Conv1d(base * 2, base * 4, 9, stride=2, padding=4),
            nn.BatchNorm1d(base * 4), activation(base * 4), block(base * 4, 9),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(base * 4, base * 8, 9, stride=2, padding=4),
            nn.BatchNorm1d(base * 8), activation(base * 8), block(base * 8, 7),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        branch_features = base * 8 * 2
        self.head = nn.Sequential(
            nn.Linear(branch_features * 2, d_model), activation(d_model),
            nn.Dropout(0.7), nn.Linear(d_model, 1),
        )

    def _encode(self, x):
        features = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return torch.cat([
            self.avg_pool(features).squeeze(-1),
            self.max_pool(features).squeeze(-1),
        ], dim=1)

    def forward(self, x):
        channels = x.shape[1] // 2
        first, second = self._encode(x[:, :channels]), self._encode(x[:, channels:])
        combined = (
            torch.cat([first * second, torch.abs(first - second)], dim=1)
            if self.use_physics_fusion else torch.cat([first, second], dim=1)
        )
        return self.head(combined)
