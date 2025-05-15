"""
    based on the paper: https://arxiv.org/pdf/1811.11286
    actual code repo: https://github.com/yifita/3PU_pytorch

    the code below is a simplification of the core ideas behind 3PU:
    1. Downsample to patches using farthest point sampling
	2. Extract features using MLPs and max pooling (like PointNet)
	3. Progressive upsampling using folding-based decoder modules
	4. No residual loss/edge loss or advanced regularizers here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    B, N, _ = x.shape
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=-1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20):
    B, N, C = x.shape
    idx = knn(x, k)
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.view(B * N, -1)
    feature = x[idx, :].view(B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature  # (B, 2C, N, k)

class PatchFeatureExtractor(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim * 2, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Conv2d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        # x: (B, N, 3)
        x = get_graph_feature(x)  # (B, 6, N, k)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=-1)[0]  # max over k neighbors -> (B, out_dim, N)
        return x

class FoldingDecoder(nn.Module):
    def __init__(self, feature_dim=128, up_ratio=2):
        super().__init__()
        self.grid_size = int(up_ratio ** 0.5)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(feature_dim + 2, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
        )

    def forward(self, global_feat):
        B, C, N = global_feat.shape
        grid = self.build_grid(N, B, global_feat.device)  # (B, 2, N * r)
        global_feat = global_feat.unsqueeze(3).repeat(1, 1, 1, self.grid_size ** 2)
        global_feat = global_feat.view(B, C, -1)
        x = torch.cat([global_feat, grid], dim=1)
        out = self.mlp1(x)
        return out

    def build_grid(self, n_points, batch_size, device):
        grid_range = torch.linspace(-0.2, 0.2, steps=self.grid_size, device=device)
        meshgrid = torch.meshgrid(grid_range, grid_range, indexing='ij')
        grid = torch.stack(meshgrid, dim=-1).reshape(-1, 2)  # (r, 2)
        grid = grid.unsqueeze(0).repeat(batch_size, n_points, 1)  # (B, N, 2)
        return grid.permute(0, 2, 1).contiguous()  # (B, 2, N*r)

class Simple3PU(nn.Module):
    def __init__(self, up_factors=[2, 2]):  # 2x then 2x → total 4x
        super().__init__()
        self.extractor = PatchFeatureExtractor()
        self.decoders = nn.ModuleList()
        for factor in up_factors:
            self.decoders.append(FoldingDecoder(feature_dim=128, up_ratio=factor))

    def forward(self, xyz):
        B, N, _ = xyz.shape
        feat = self.extractor(xyz)  # (B, 128, N)
        for decoder in self.decoders:
            xyz_offset = decoder(feat)  # (B, 3, N * up_ratio)
            xyz = xyz_offset.permute(0, 2, 1).contiguous()
            feat = self.extractor(xyz)  # update feature for next stage
        return xyz


