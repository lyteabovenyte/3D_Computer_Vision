"""
    Implementation of PointINet for point cloud frame interpolation based on the paper: https://arxiv.org/pdf/2012.10066

    this code is based on the official implementation of PointINet which involves:
    1. Point Feature Extraction Module
    2. Cross-Frame Feature Interaction Module
    3. Point Generation Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Helper functions for point cloud operations
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def ball_query(xyz, new_xyz, radius, nsample):
    """
    Input:
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, M, 3]
        radius: search radius
        nsample: max number of points in each ball
    Return:
        group_idx: indices of points in each ball, [B, M, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])
    sqrdists = torch.sum((xyz.unsqueeze(2) - new_xyz.unsqueeze(1)) ** 2, -1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_idx[group_idx == N] = 0  # Replace invalid indices
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, nsample]
    Return:
        new_points: indexed points data, [B, S, nsample, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

# Set Abstraction Layer
class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, npoint, 3]
            new_points: sampled points data, [B, npoint, D]
        """
        B, N, C = xyz.shape
        # Farthest point sampling
        idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, idx)
        # Ball query
        group_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)
        grouped_xyz = index_points(xyz, group_idx)  # [B, npoint, nsample, 3]
        grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)
        if points is not None:
            grouped_points = index_points(points, group_idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        # PointNet encoding
        grouped_points = grouped_points.permute(0, 3, 1, 2)  # [B, C, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))
        new_points = torch.max(grouped_points, 3)[0]  # [B, D, npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, D]
        return new_xyz, new_points

# Feature Propagation Layer
class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled points position data, [B, M, 3]
            points1: input points data, [B, N, D1]
            points2: sampled points data, [B, M, D2]
        Return:
            new_points: upsampled points data, [B, N, D]
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        if M == 1:
            return points2.repeat(1, N, 1)
        dists = torch.sum((xyz1.unsqueeze(2) - xyz2.unsqueeze(1)) ** 2, -1)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # Take 3 nearest points
        dists = torch.clamp(dists, min=1e-10)
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1, keepdim=True)
        interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)  # [B, D, N]
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
        new_points = new_points.permute(0, 2, 1)  # [B, N, D]
        return new_points

# PointINet Architecture
class PointINet(nn.Module):
    def __init__(self, num_points=2048, feature_dim=1):
        super(PointINet, self).__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim

        # Point Feature Extraction Module
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + feature_dim, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = SetAbstraction(npoint=1, radius=0.8, nsample=128, in_channel=256, mlp=[256, 256, 512])
        self.fp3 = FeaturePropagation(in_channel=256 + 512, mlp=[256, 256])
        self.fp2 = FeaturePropagation(in_channel=128 + 256, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128 + (3 + feature_dim), mlp=[128, 128])

        # Cross-Frame Feature Interaction Module
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Point Generation Module
        self.coord_mlp = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1)
        )
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feature_dim, 1)
        )

    def forward(self, pc1, pc2, tau):
        """
        Input:
            pc1: point cloud at t, [B, N, 3 + C]
            pc2: point cloud at t+1, [B, N, 3 + C]
            tau: interpolation parameter, scalar or [B]
        Return:
            pc_interp: interpolated point cloud, [B, N, 3 + C]
        """
        B, N, C = pc1.shape
        xyz1, feat1 = pc1[:, :, :3], pc1[:, :, 3:]
        xyz2, feat2 = pc2[:, :, :3], pc2[:, :, 3:]

        # Point Feature Extraction for pc1
        l1_xyz1, l1_points1 = self.sa1(xyz1, feat1)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)
        global_feat1 = l3_points1.squeeze(1)  # [B, 512]
        l2_points1 = self.fp3(l2_xyz1, l3_xyz1, l2_points1, l3_points1)
        l1_points1 = self.fp2(l1_xyz1, l2_xyz1, l1_points1, l2_points1)
        points1 = self.fp1(xyz1, l1_xyz1, pc1, l1_points1)  # [B, N, 128]

        # Point Feature Extraction for pc2
        l1_xyz2, l1_points2 = self.sa1(xyz2, feat2)
        l2_xyz2, l2_points2 = self.sa2(l1_xyz2, l1_points2)
        l3_xyz2, l3_points2 = self.sa3(l2_xyz2, l2_points2)
        global_feat2 = l3_points2.squeeze(1)  # [B, 512]
        l2_points2 = self.fp3(l2_xyz2, l3_xyz2, l2_points2, l3_points2)
        l1_points2 = self.fp2(l1_xyz2, l2_xyz2, l1_points2, l2_points2)
        points2 = self.fp1(xyz2, l1_xyz2, pc2, l1_points2)  # [B, N, 128]

        # Cross-Frame Feature Interaction
        # Concatenate features
        combined_points = torch.cat([points1, points2], dim=1)  # [B, 2N, 128]
        # Attention
        attn_output, _ = self.attention(combined_points, combined_points, combined_points)
        points1_attn, points2_attn = attn_output.chunk(2, dim=1)  # [B, N, 128] each
        # Temporal fusion
        tau = tau.view(B, 1, 1) if tau.dim() == 1 else tau
        fused_points = (1 - tau) * points1_attn + tau * points2_attn
        global_fused = (1 - tau.squeeze()) * global_feat1 + tau.squeeze() * global_feat2
        # Refine fused features
        global_fused_exp = global_fused.unsqueeze(1).repeat(1, N, 1)  # [B, N, 512]
        fused_points = torch.cat([fused_points, global_fused_exp], dim=-1).permute(0, 2, 1)
        fused_points = self.fusion_mlp(fused_points).permute(0, 2, 1)  # [B, N, 128]

        # Point Generation
        gen_input = torch.cat([fused_points, global_fused_exp], dim=-1).permute(0, 2, 1)
        delta_xyz = self.coord_mlp(gen_input).permute(0, 2, 1)  # [B, N, 3]
        interp_xyz = xyz1 + tau * delta_xyz
        interp_feat = self.feature_mlp(gen_input).permute(0, 2, 1)  # [B, N, C]
        pc_interp = torch.cat([interp_xyz, interp_feat], dim=-1)  # [B, N, 3 + C]
        return pc_interp

# Chamfer Distance Loss
def chamfer_distance(pred, gt):
    """
    Input:
        pred: predicted points, [B, N, 3]
        gt: ground truth points, [B, N, 3]
    Return:
        loss: Chamfer distance
    """
    dist1 = torch.sum((pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2, dim=-1)
    dist2 = torch.sum((gt.unsqueeze(2) - pred.unsqueeze(1)) ** 2, dim=-1)
    loss1 = torch.mean(torch.min(dist1, dim=2)[0])
    loss2 = torch.mean(torch.min(dist2, dim=2)[0])
    return loss1 + loss2

# Example Training Loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointINet(num_points=2048, feature_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    B, N = 8, 2048
    pc1 = torch.rand(B, N, 4).to(device)  # [B, N, 3 + 1]
    pc2 = torch.rand(B, N, 4).to(device)
    pc_gt = torch.rand(B, N, 4).to(device)  # Ground truth interpolated frame
    tau = torch.tensor(0.5).to(device)
    
    model.training = True
    for epoch in range(100):
        optimizer.zero_grad()
        pc_pred = model(pc1, pc2, tau)
        coord_loss = chamfer_distance(pc_pred[:, :, :3], pc_gt[:, :, :3])
        feat_loss = F.mse_loss(pc_pred[:, :, 3:], pc_gt[:, :, 3:])
        loss = coord_loss + 0.1 * feat_loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
