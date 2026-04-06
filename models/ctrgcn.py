"""
models/ctrgcn.py
Channel-wise Topology Refinement GCN (CTR-GCN)
Paper: Chen et al., ICCV 2021

Cải tiến so với ST-GCN:
- Topology graph được học riêng cho từng channel (channel-wise)
- Dùng cả static topology (A) lẫn dynamic topology (học từ data)

Input : (B, C=3, T=64, J=75)
Output: (B, n_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelTopologyRefinement(nn.Module):
    """
    Học dynamic adjacency matrix riêng cho từng channel group.
    Kết hợp static A và dynamic A_learned.
    """
    def __init__(self, in_channels: int, n_joints: int, n_groups: int = 8):
        super().__init__()
        self.n_groups = n_groups
        ch_per_group  = in_channels // n_groups

        # 2 nhánh để học dynamic topology
        self.conv_a = nn.Conv2d(in_channels, n_groups * n_joints, kernel_size=1)
        self.conv_b = nn.Conv2d(in_channels, n_groups * n_joints, kernel_size=1)

        self.alpha  = nn.Parameter(torch.zeros(1))  # learned weight

    def forward(self, x: torch.Tensor, A_static: torch.Tensor):
        """
        x       : (B, C, T, J)
        A_static: (J, J)
        Returns : (B, C, T, J) — sau graph conv với refined topology
        """
        B, C, T, J = x.shape

        # Dynamic topology: outer product of two projections
        qa = self.conv_a(x).mean(dim=2)       # (B, G*J, J_src) → mean over T
        qb = self.conv_b(x).mean(dim=2)       # (B, G*J, J_src)
        qa = qa.view(B, self.n_groups, J, J)
        qb = qb.view(B, self.n_groups, J, J)
        A_dyn = torch.softmax(torch.bmm(
            qa.view(B * self.n_groups, J, J),
            qb.view(B * self.n_groups, J, J).transpose(1, 2)
        ), dim=-1).view(B, self.n_groups, J, J)  # (B, G, J, J)

        # Combine static + dynamic
        A_refined = A_static.unsqueeze(0).unsqueeze(0) + self.alpha * A_dyn
        # A_refined: (B, G, J, J)

        # Split channels to groups, apply per-group adjacency
        x_groups = x.view(B, self.n_groups, C // self.n_groups, T, J)
        out_groups = torch.einsum('bgctv,bgvw->bgctw', x_groups, A_refined)
        return out_groups.view(B, C, T, J)


class CTRGCNBlock(nn.Module):
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 A:            torch.Tensor,
                 stride:       int   = 1,
                 dropout:      float = 0.0,
                 n_groups:     int   = 8):
        super().__init__()
        self.register_buffer('A', A)

        # Spatial: CTR
        self.ctr = ChannelTopologyRefinement(in_channels, A.shape[0], n_groups)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Temporal
        t_pad = (9 - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(9, 1),
                      stride=(stride, 1),
                      padding=(t_pad, 0)),
            nn.BatchNorm2d(out_channels),
        )

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
        x_ref = self.ctr(x, self.A)
        x_gcn = self.gcn(x_ref)
        out   = self.tcn(x_gcn) + self.residual(x)
        return self.relu(self.dropout(out))


class CTRGCN(nn.Module):
    def __init__(self,
                 A:           np.ndarray,
                 n_classes:   int,
                 in_channels: int   = 3,
                 dropout:     float = 0.3):
        super().__init__()
        A_tensor = torch.tensor(A, dtype=torch.float32)

        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[0])

        config = [
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
            # n_groups phải chia hết in_ch — dùng min
            n_groups = min(8, in_ch)
            while in_ch % n_groups != 0:
                n_groups -= 1
            self.blocks.append(
                CTRGCNBlock(in_ch, out_ch, A_tensor,
                            stride=stride, dropout=dropout, n_groups=n_groups)
            )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, J = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, J * C, T)
        x = self.data_bn(x)
        x = x.view(B, J, C, T).permute(0, 2, 3, 1)

        for blk in self.blocks:
            x = blk(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)
