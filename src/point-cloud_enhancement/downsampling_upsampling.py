"""
    downsampling and upsampling using Farthest Point Sampling (FPS) and Inverse Distance Weighting (IDW)
    based on PointNet++ paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

# -----------------------
# Farthest Point Sampling
# -----------------------
def farthest_point_sampling(xyz, npoint):
    """
    Input:
        xyz: (B, N, 3) point cloud
        npoint: number of points to sample
    Return:
        centroids_idx: (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)

    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


# -----------------------
# Feature Interpolation
# -----------------------
def interpolate_features(xyz_src, xyz_dst, feat_src, k=3):
    """
    Interpolate features from source points to destination points.
    
    xyz_src: (B, N1, 3) downsampled points
    xyz_dst: (B, N2, 3) original dense points
    feat_src: (B, N1, C) features from sparse points
    Returns:
        interpolated features: (B, N2, C)
    """
    B, N2, _ = xyz_dst.shape
    N1 = xyz_src.shape[1]

    # Flatten batch
    xyz_src_flat = xyz_src.reshape(B * N1, 3)
    xyz_dst_flat = xyz_dst.reshape(B * N2, 3)

    # kNN: find k nearest neighbors in src for each dst point
    idx = knn(xyz_src_flat, xyz_dst_flat, k=k, batch_x=torch.arange(B).repeat_interleave(N2),
              batch_y=torch.arange(B).repeat_interleave(N1))

    # idx: (2, B * N2 * k) — [dst_idx, src_idx]
    dst_idx, src_idx = idx

    # Gather features and positions
    feat_src_flat = feat_src.reshape(B * N1, -1)
    xyz_src_knn = xyz_src_flat[src_idx]  # (B*N2*k, 3)
    xyz_dst_knn = xyz_dst_flat[dst_idx]  # (B*N2*k, 3)
    feat_knn = feat_src_flat[src_idx]    # (B*N2*k, C)

    # Inverse distance weights
    dist = torch.norm(xyz_dst_knn - xyz_src_knn, dim=1) + 1e-10
    weight = 1.0 / dist
    weight = weight / torch_scatter.scatter_add(weight, dst_idx, dim=0)[dst_idx]

    weighted_feat = feat_knn * weight.unsqueeze(1)
    feat_out = torch_scatter.scatter_add(weighted_feat, dst_idx, dim=0, dim_size=B * N2)

    feat_out = feat_out.view(B, N2, -1)
    return feat_out


# -----------------------
# Example Usage
# -----------------------

class PointNetFPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, xyz_dense, xyz_sparse, feat_sparse):
        feat_interp = interpolate_features(xyz_sparse, xyz_dense, feat_sparse)
        return self.mlp(feat_interp)

# Simulate a small test
if __name__ == '__main__':
    B, N, C = 2, 1024, 64
    D = 256  # downsampled points

    xyz = torch.rand(B, N, 3).cuda()
    feats = torch.rand(B, N, C).cuda()

    # Downsample
    sampled_idx = farthest_point_sampling(xyz, D)  # (B, D)
    xyz_down = torch.gather(xyz, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
    feat_down = torch.gather(feats, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, C))

    # Feature Propagation
    fp_block = PointNetFPBlock(in_channels=C, out_channels=128).cuda()
    feat_upsampled = fp_block(xyz, xyz_down, feat_down)

    print("Upsampled feature shape:", feat_upsampled.shape)  # (B, N, 128)