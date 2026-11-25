
"""Residual 1D CNN for multi-label ECG classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock1D(nn.Module):
    """Residual convolutional block with an optional projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if min(in_channels, out_channels, stride) <= 0:
            raise ValueError("Channel dimensions and stride must be positive.")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        return self.activation(out)


class ECGResNet1D(nn.Module):
    """Residual 1D CNN for five-class multi-label ECG classification."""

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive.")

        if not channels or len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have equal nonzero length.")

        if any(channel <= 0 for channel in channels):
            raise ValueError("All channel dimensions must be positive.")

        if any(count <= 0 for count in blocks_per_stage):
            raise ValueError("Each stage must contain at least one residual block.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                channels[0],
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )

        stages: list[nn.Module] = []
        current_channels = channels[0]

        for stage_index, (out_channels, num_blocks) in enumerate(
            zip(channels, blocks_per_stage, strict=True)
        ):
            blocks: list[nn.Module] = []

            for block_index in range(num_blocks):
                stride = 2 if stage_index > 0 and block_index == 0 else 1

                blocks.append(
                    ResidualBlock1D(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        stride=stride,
                        kernel_size=7,
                        dropout=dropout / 2,
                    )
                )
                current_channels = out_channels

            stages.append(nn.Sequential(*blocks))

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
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

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )

        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)

        return self.classifier(x)
