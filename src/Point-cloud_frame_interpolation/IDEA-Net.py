"""
    Interpolation via Deep Embedding Alignment Network (IDEA-Net): https://arxiv.org/pdf/2203.11590

    the core innovation of the IDEA-Net is the deep embedding alignment network, which is a novel network architecture
    that can learn the correspondence between two point clouds in a self-supervised manner.

    The architecture operates through a two-step coarse-to-fine process: 
    coarse linear interpolation followed by trajectory compensation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_points3d.metrics.chamfer_distance import ChamferDistance
from einops import rearrange
import math

class PointNetFeatureExtractor(nn.Module):
    """PointNet++-inspired feature extractor for point clouds."""
    def __init__(self, in_dim=3, out_dim=256):
        super(PointNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # (B, out_dim, N) -> (B, N, out_dim)
        x = x.transpose(1, 2)
        return x

class CrossAttentionModule(nn.Module):
    """Transformer-based cross-attention for embedding alignment (The Novel idea of IDEA-Net)."""
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super(CrossAttentionModule, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, f0, f1):
        # f0, f1: (B, N, dim)
        B, N, _ = f0.shape

        # Compute Q, K, V
        q = self.q_linear(f0)  # (B, N, dim)
        k = self.k_linear(f1)  # (B, N, dim)
        v = self.v_linear(f1)  # (B, N, dim)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)

        # Scaled dot-product attention
        scores = torch.einsum('bhnd,bhmd->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Output projection and residual connection
        out = self.out_linear(out)
        out = self.layer_norm(f0 + out)  # Residual connection
        return out

class InterpolationModule(nn.Module):
    """Interpolates aligned features and decodes to 3D points."""
    def __init__(self, in_dim=256, out_dim=3):
        super(InterpolationModule, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, f0_aligned, f1_aligned, t):
        # f0_aligned, f1_aligned: (B, N, dim)
        # t: scalar in [0, 1]
        ft = (1 - t) * f0_aligned + t * f1_aligned
        pt = self.decoder(ft)  # (B, N, 3)
        return pt

class IDEANet(nn.Module):
    """IDEA-Net for dynamic 3D point cloud interpolation."""
    def __init__(self, feature_dim=256):
        super(IDEANet, self).__init__()
        self.feature_extractor = PointNetFeatureExtractor(in_dim=3, out_dim=feature_dim)
        self.cross_attention = CrossAttentionModule(dim=feature_dim)
        self.interpolation_module = InterpolationModule(in_dim=feature_dim, out_dim=3)
        self.chamfer_dist = ChamferDistance()

    def forward(self, p0, p1, t):
        # p0, p1: (B, N, 3)
        # t: scalar in [0, 1]
        
        # Extract features
        f0 = self.feature_extractor(p0)  # (B, N, feature_dim)
        f1 = self.feature_extractor(p1)  # (B, N, feature_dim)

        # Align embeddings
        f0_aligned = self.cross_attention(f0, f1)  # f0 attends to f1
        f1_aligned = self.cross_attention(f1, f0)  # f1 attends to f0

        # Interpolate and generate points
        pt = self.interpolation_module(f0_aligned, f1_aligned, t)  # (B, N, 3)
        return pt, f0_aligned, f1_aligned

    def compute_loss(self, p0, p1, pt_pred, pt_gt=None, f0_aligned=None, f1_aligned=None):
        """Compute loss: Chamfer Distance, cycle consistency, and smoothness."""
        losses = {}

        # Reconstruction loss (if ground truth is available)
        if pt_gt is not None:
            dist1, dist2 = self.chamfer_dist(pt_pred, pt_gt)
            losses['recon'] = (dist1.mean() + dist2.mean()) / 2

        # Cycle consistency: Interpolate back to p0 and p1
        pt0_pred = self.interpolation_module(f0_aligned, f1_aligned, t=0.0)
        pt1_pred = self.interpolation_module(f0_aligned, f1_aligned, t=1.0)
        dist1_p0, dist2_p0 = self.chamfer_dist(pt0_pred, p0)
        dist1_p1, dist2_p1 = self.chamfer_dist(pt1_pred, p1)
        losses['cycle'] = (dist1_p0.mean() + dist2_p0.mean() + dist1_p1.mean() + dist2_p1.mean()) / 4

        # Smoothness loss: Penalize acceleration
        if pt_gt is not None:
            # Approximate second derivative using finite differences
            dt = 0.1
            pt_pred_t1 = self.interpolation_module(f0_aligned, f1_aligned, t=min(t+dt, 1.0))
            pt_pred_t2 = self.interpolation_module(f0_aligned, f1_aligned, t=max(t-dt, 0.0))
            accel = (pt_pred_t1 - 2 * pt_pred + pt_pred_t2) / (dt ** 2)
            losses['smooth'] = torch.mean(accel ** 2)

        # Total loss
        weights = {'recon': 1.0, 'cycle': 0.5, 'smooth': 0.1}
        total_loss = sum(w * losses[k] for k, w in weights.items() if k in losses)
        return total_loss, losses

# Example usage
if __name__ == "__main__":
    # Dummy data
    batch_size, num_points = 4, 2048
    p0 = torch.rand(batch_size, num_points, 3).cuda()
    p1 = torch.rand(batch_size, num_points, 3).cuda()
    t = 0.5
    pt_gt = torch.rand(batch_size, num_points, 3).cuda()  # Ground truth (optional)

    # Initialize model
    model = IDEANet(feature_dim=256).cuda()
    model.train()

    # Forward pass
    pt_pred, f0_aligned, f1_aligned = model(p0, p1, t)

    # Compute loss
    total_loss, losses = model.compute_loss(p0, p1, pt_pred, pt_gt, f0_aligned, f1_aligned)
    print(f"Total Loss: {total_loss.item()}")
    print(f"Loss Breakdown: {losses}")