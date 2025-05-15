import torch
import torch.nn as nn

def farthest_point_sampling(xyz, npoint):
    """
    Input:
        xyz: (B, N, 3)
        npoint: int
    Return:
        centroids: (B, npoint) indices of centroids
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    
    return centroids

def ball_query(xyz, new_xyz, radius, nsample):
    """
    Input:
        xyz: (B, N, 3)
        new_xyz: (B, S, 3)
    Return:
        group_idx: (B, S, nsample)
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((xyz.unsqueeze(1) - new_xyz.unsqueeze(2)) ** 2, dim=-1)
    group_idx[sqrdists > radius ** 2] = N  # Mask out-of-radius points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx[group_idx == N] = group_first[group_idx == N]
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: (B, N, C)
        idx: (B, S, nsample)
    Return:
        new_points: (B, S, nsample, C)
    """
    B = points.shape[0]
    S, nsample = idx.shape[1], idx.shape[2]
    batch_indices = torch.arange(B).view(B, 1, 1).repeat(1, S, nsample)
    return points[batch_indices, idx]

class SetConv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample):
        super(SetConv, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 1),
            nn.ReLU()
        )

    def forward(self, xyz, points, npoint):
        """
        Input:
            xyz: (B, N, 3)
            points: (B, N, C)
        Output:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, out_channels)
        """
        B, N, _ = xyz.shape

        # 1. Sample points
        idx = farthest_point_sampling(xyz, npoint)  # (B, npoint)
        new_xyz = index_points(xyz, idx.unsqueeze(-1)).squeeze(2)  # (B, npoint, 3)

        # 2. Group neighbors
        group_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B, npoint, nsample)
        grouped_xyz = index_points(xyz, group_idx)  # (B, npoint, nsample, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, group_idx)  # (B, npoint, nsample, C)
            new_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, nsample, C+3)
        else:
            new_features = grouped_xyz_norm  # (B, npoint, nsample, 3)

        new_features = new_features.permute(0, 3, 1, 2)  # (B, C+3, npoint, nsample)
        new_points = self.mlp(new_features)  # (B, out_channels, npoint, nsample)
        new_points = torch.max(new_points, dim=-1)[0]  # (B, out_channels, npoint)

        return new_xyz, new_points.permute(0, 2, 1)  # (B, npoint, 3), (B, npoint, out_channels)