"""
models/stgcn.py
Spatial-Temporal Graph Convolutional Network (ST-GCN)
Paper: Yan et al., AAAI 2018

Input : (B, C=3, T=64, J=75)
Output: (B, n_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STGCNBlock(nn.Module):
    """1 khối ST-GCN = Spatial GCN + Temporal Conv + Residual"""

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 A:            torch.Tensor,
                 stride:       int = 1,
                 dropout:      float = 0.0):
        super().__init__()
        self.register_buffer('A', A)  # (J, J)

        # Spatial: Graph Conv (point-wise conv × adjacency)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Temporal: Depthwise Conv trên axis T
        temporal_pad = (9 - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(9, 1),
                      stride=(stride, 1),
                      padding=(temporal_pad, 0)),
            nn.BatchNorm2d(out_channels),
        )

        # Residual
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J)
        # Spatial graph conv: x @ A^T
        x_gcn = torch.einsum('nctv,vw->nctw', x, self.A)
        x_gcn = self.gcn(x_gcn)

        # Temporal conv
        out = self.tcn(x_gcn) + self.residual(x)
        out = self.dropout(out)
        return self.relu(out)


class STGCN(nn.Module):
    def __init__(self,
                 A:          np.ndarray,
                 n_classes:  int,
                 in_channels: int   = 3,
                 dropout:     float = 0.3):
        super().__init__()
        A_tensor = torch.tensor(A, dtype=torch.float32)

        # Input BN
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[0])

        # 10 ST-GCN blocks (giống bản gốc)
        config = [
            # (in_ch, out_ch, stride)
            (in_channels, 64,  1),
            (64,          64,  1),
            (64,          64,  1),
            (64,          64,  1),
            (64,          128, 2),
            (128,         128, 1),
            (128,         128, 1),
            (128,         256, 2),
            (256,         256, 1),
            (256,         256, 1),
        ]

        self.blocks = nn.ModuleList()
        for in_ch, out_ch, stride in config:
            self.blocks.append(
                STGCNBlock(in_ch, out_ch, A_tensor, stride=stride, dropout=dropout)
            )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J)
        B, C, T, J = x.shape

        # Input batch norm
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, J, C, T)
        x = x.view(B, J * C, T)
        x = self.data_bn(x)
        x = x.view(B, J, C, T).permute(0, 2, 3, 1)  # (B, C, T, J)

        for blk in self.blocks:
            x = blk(x)

        x = self.pool(x)   # (B, 256, 1, 1)
        x = x.flatten(1)   # (B, 256)
        return self.fc(x)  # (B, n_classes)
