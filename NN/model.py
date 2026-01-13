"""
GeoFNO: Geometric Fourier Neural Operator
"""

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    """
    Two-dimensional Spectral Convolution layer.
    Computes FFT, applies complex weight multiplication in Fourier space,
    and returns to physical space using Inverse FFT.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, input_tensor, weights):
        """Complex multiplication in Fourier space."""
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x):
        """
        Forward pass for SpectralConv2d.
        """
        batchsize = x.shape[0]
        # Compute multi-dimensional real-valued FFT
        x_ft = torch.fft.rfft2(x)

        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Multiply relevant Fourier modes
        # Upper block
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # Lower block
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    """
    Fourier Neural Operator Block.
    Combines a spectral convolution with a regular 1x1 convolution
    and a residual connection.
    Includes Instance Normalization for improved training stability.
    """

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Forward pass for FNOBlock.
        """
        x1 = self.conv(x)
        x2 = self.w(x)
        return self.act(self.norm(x1 + x2)) + x


# pylint: disable=too-many-instance-attributes
class DecoderHead(nn.Module):
    """
    Modular decoder head for coordinate injection MLP.
    Used for separate temperature and velocity predictions.
    """

    def __init__(self, in_features, hidden_dim, out_features, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward pass for decoder head."""
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class GeoFNO(nn.Module):
    """
    Geometric Fourier Neural Operator (GeoFNO).
    Extends FNO by utilizing coordinate injection in the decoder to handle
    geometric queries and irregular mesh predictions.

    Uses separate decoder heads for temperature and velocity.
    """

    def __init__(self, modes1, modes2, width, dropout_rate):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Encoder: Projects the multi-channel input to latent space
        # Input channels: conductivity, power, x, y, mask, vel_inlet_x, vel_inlet_y
        self.fc0 = nn.Conv2d(7, self.width, 1)

        # Body: Sequential FNO layers
        self.fno_blocks = nn.ModuleList(
            [FNOBlock(self.width, self.modes1, self.modes2) for _ in range(4)]
        )

        # Separate Decoder Heads with Coordinate Injection
        # Input: Width (latent features) + 2 (physical X,Y coordinates)
        decoder_in = self.width + 2
        self.decoder_T = DecoderHead(decoder_in, 128, 1, dropout_rate)  # Temperature
        self.decoder_V = DecoderHead(decoder_in, 128, 2, dropout_rate)  # Velocity (vx, vy)

    def forward(self, x_grid, query_coords):
        """
        Forward pass for GeoFNO.

        Args:
            x_grid: Input grid tensor of shape (B, C, H, W)
            query_coords: Query coordinates on the mesh (B, N, 2)

        Returns:
            Predicted values at query coordinates (B, N, 3) - [T, vx, vy]
        """
        # 1. Grid Encoding
        x = self.fc0(x_grid)

        # 2. Spectral Processing
        for block in self.fno_blocks:
            x = block(x)

        # 3. Geometric Querying (Sampling from grid to mesh)
        # Reshape query_coords for F.grid_sample: (B, 1, N, 2)
        q = query_coords.unsqueeze(1)

        # Interpolate spectral features onto mesh coordinates
        x_sampled = F.grid_sample(
            x, q, mode="bilinear", padding_mode="border", align_corners=False
        )

        # Final shape: (B, N, width)
        x_sampled = x_sampled.squeeze(2).permute(0, 2, 1)

        # 4. Coordinate Injection
        # Concatenate latent features with exact physical coordinates
        x_cat = torch.cat([x_sampled, query_coords], dim=-1)

        # 5. Point-wise Decoding with Separate Heads
        out_T = self.decoder_T(x_cat)  # (B, N, 1)
        out_V = self.decoder_V(x_cat)  # (B, N, 2)

        # Combine outputs: T, vx, vy
        out = torch.cat([out_T, out_V], dim=-1)

        return out

