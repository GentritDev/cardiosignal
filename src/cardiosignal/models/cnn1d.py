
"""1D CNN baseline for multi-label ECG classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock1D(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout1d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ECG1DCNN(nn.Module):
    """Baseline 1D CNN for five-class multi-label ECG classification.

    Expected input:
        (batch_size, 12, 5000)

    Output:
        (batch_size, 5) raw logits.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive.")

        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive values.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = []
        current_channels = in_channels

        for index, out_channels in enumerate(channels):
            layers.append(
                ConvBlock1D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=7 if index == 0 else 5,
                    stride=2,
                    dropout=dropout / 2,
                )
            )
            current_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(current_channels, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D input (batch, channels, time), got {tuple(x.shape)}."
            )

        if x.shape[1] != self.feature_extractor[0].block[0].in_channels:
            raise ValueError(
                f"Expected {self.feature_extractor[0].block[0].in_channels} "
                f"input channels, got {x.shape[1]}."
            )

        x = self.feature_extractor(x)
        x = self.pool(x)
        return self.classifier(x)
