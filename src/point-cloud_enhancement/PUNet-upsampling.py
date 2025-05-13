import torch
import torch.nn as nn
import torch.nn.functional as F


class PUFeatureExpansion(nn.Module):
    """
    PU-Net Feature Expansion Module for Point Cloud Upsampling
    Upsamples the input point features by the given ratio.
    """

    def __init__(self, in_channels: int, out_channels: int, up_ratio: int):
        """
        Args:
            in_channels (int): Number of input feature channels
            out_channels (int): Number of output feature channels before final coord regressor
            up_ratio (int): Upsampling ratio (e.g., 4)
        """
        super(PUFeatureExpansion, self).__init__()
        self.up_ratio = up_ratio

        # Multi-branch shared MLP layers
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 1),
                nn.ReLU()
            ) for _ in range(up_ratio)
        ])

    def forward(self, x):
        # x: (B, C_in, N)
        B, C, N = x.size()
        out = []

        for i in range(self.up_ratio):
            branch_out = self.branches[i](x)  # (B, C_out, N)
            out.append(branch_out)

        # Concatenate on point dimension
        out = torch.cat(out, dim=2)  # (B, C_out, r*N)
        return out


class PUCoordRegressor(nn.Module):
    """
    MLP that projects expanded features into 3D coordinates.
    """

    def __init__(self, in_channels: int):
        super(PUCoordRegressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels, 3, 1)
        )

    def forward(self, x):
        # x: (B, C, r*N)
        coords = self.mlp(x)  # (B, 3, r*N)
        return coords


class PUUpsampler(nn.Module):
    """
    Full PU-Net Upsampling Block: Expansion + Coordinate Regression
    """

    def __init__(self, in_channels: int, mid_channels: int, up_ratio: int):
        super(PUUpsampler, self).__init__()
        self.feature_expand = PUFeatureExpansion(in_channels, mid_channels, up_ratio)
        self.coord_regressor = PUCoordRegressor(mid_channels)

    def forward(self, x):
        """
        Args:
            x: input point features, shape (B, C_in, N)
        Returns:
            upsampled points: (B, 3, r*N)
        """
        expanded_feat = self.feature_expand(x)  # (B, C_mid, r*N)
        coords = self.coord_regressor(expanded_feat)  # (B, 3, r*N)
        return coords