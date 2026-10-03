import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + self.res_scale * residual


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()

        layers = []

        # ×4 = ×2 followed by ×2
        for _ in range(scale // 2):
            layers.append(
                nn.Conv2d(
                    channels,
                    channels * 4,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.PixelShuffle(2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EDSR(nn.Module):
    def __init__(
        self,
        channels=64,
        num_blocks=16,
        scale=4
    ):
        super().__init__()

        # First feature extraction
        self.head = nn.Conv2d(
            3,
            channels,
            kernel_size=3,
            padding=1
        )

        # Residual blocks
        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Body reconstruction
        self.body_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # Upscaling
        self.upsample = UpsampleBlock(
            channels,
            scale
        )

        # Final RGB reconstruction
        self.tail = nn.Conv2d(
            channels,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Shallow features
        x = self.head(x)
        skip = x

        # Deep features
        x = self.body(x)
        x = self.body_conv(x)

        # Global residual connection
        x = x + skip

        # ×4 upsampling
        x = self.upsample(x)

        # RGB output
        x = self.tail(x)

        return x