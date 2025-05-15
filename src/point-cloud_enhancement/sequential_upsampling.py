"""
    Sequential upsampling of point clouds can be used for video frame interpolation or temporal upsampling
    which is used to upsample points in motion.

    the breakthrough:
    phase 1. Feature Extraction

    Each sparse point cloud (e.g., at t-1, t, t+1) is passed through a shared point-wise feature extractor, typically a PointNet or PointNet++ encoder, to compute high-dimensional features per point.
	- Input: Sparse point cloud $\mathbf{P}_t \in \mathbb{R}^{N \times 3}$
	- Output: Feature map $\mathbf{F}_t \in \mathbb{R}^{N \times C}$

    phase 2. Temporal Alignment

    Since points at different timestamps are not aligned spatially due to motion, the temporal alignment block warps or aligns features across time.

    Typical implementations:
	- Flow-based warping (scene flow, optical flow)
	- Learned alignment via attention or deformable modules
	- Input: Feature maps at t-1, t+1
	- Output: Temporally aligned features with respect to t

    phase 3. Feature Aggregation

    Once aligned, features from all time steps are aggregated, usually by concatenation or attention fusion.

    phase 4. Upsampling

    Finally, the aggregated feature is used to predict dense point clouds via upsampling networks (e.g., PU-GAN, FoldingNet, etc.).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

#! PointNet can be used too.
class PointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: (B, N, 3)
        return self.mlp(x)  # (B, N, output_dim)
    

class TemporalAlignment(nn.Module):
    """learnable offset between time frames"""
    def __init__(self, feature_dim):
        super().__init__()
        self.offset_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, feat_t, feat_adj):
        offset = self.offset_predictor(feat_adj)  # (B, N, C)
        return feat_adj + offset  # aligned to feat_t
    

class FeatureAggregator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(3 * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, feats):
        # feats = [feat_t1, feat_t, feat_t2], all (B, N, C)
        fused = torch.cat(feats, dim=-1)  # (B, N, 3C)
        return self.fusion(fused)  # (B, N, C)
    

# PU-GAN or PU-Transformer can be used for upsampling
class UpsampleHead(nn.Module):
    def __init__(self, input_dim, up_factor=4):
        super().__init__()
        self.up_factor = up_factor
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * up_factor)  # Predict new points
        )

    def forward(self, fused_features):
        B, N, C = fused_features.shape
        upsampled = self.mlp(fused_features)  # (B, N, 3*up_factor)
        upsampled = upsampled.view(B, N * self.up_factor, 3)  # (B, N*up_factor, 3)
        return upsampled
    

class TemporalUpsamplingModel(nn.Module):
    def __init__(self, feature_dim=128, up_factor=4):
        super().__init__()
        self.feature_extractor = PointNetFeatureExtractor(3, feature_dim)
        self.temporal_alignment = TemporalAlignment(feature_dim)
        self.feature_aggregator = FeatureAggregator(feature_dim)
        self.upsample_head = UpsampleHead(feature_dim, up_factor)

    def forward(self, pc_t_minus_1, pc_t, pc_t_plus_1):
        # Input: (B, N, 3)
        feat_t1 = self.feature_extractor(pc_t_minus_1)
        feat_t = self.feature_extractor(pc_t)
        feat_t2 = self.feature_extractor(pc_t_plus_1)

        aligned_t1 = self.temporal_alignment(feat_t, feat_t1)
        aligned_t2 = self.temporal_alignment(feat_t, feat_t2)

        fused = self.feature_aggregator([aligned_t1, feat_t, aligned_t2])
        dense_pc = self.upsample_head(fused)

        return dense_pc  # (B, N*up_factor, 3)
    
# synthetic dataset
def generate_synthetic_pc(n_points=1024, noise=0.01):
    theta = np.random.rand(n_points) * 2 * np.pi
    phi = np.random.rand(n_points) * np.pi
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=1)
    points += np.random.normal(scale=noise, size=points.shape)
    return points.astype(np.float32)

class SyntheticPointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, sparse_points=1024, dense_points=4096):
        self.num_samples = num_samples
        self.sparse_points = sparse_points
        self.dense_points = dense_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate slight motion between frames
        pc_t_minus_1 = generate_synthetic_pc(self.sparse_points)
        pc_t = generate_synthetic_pc(self.sparse_points)
        pc_t_plus_1 = generate_synthetic_pc(self.sparse_points)
        dense_gt = generate_synthetic_pc(self.dense_points)
        
        return {
            'pc_t_minus_1': torch.tensor(pc_t_minus_1),
            'pc_t': torch.tensor(pc_t),
            'pc_t_plus_1': torch.tensor(pc_t_plus_1),
            'gt_dense': torch.tensor(dense_gt)
        }
    

def chamfer_distance(p1, p2):
    """Compute Chamfer Distance between two point clouds.
    Args:
        p1: Point cloud 1 (B, N, 3)
        p2: Point cloud 2 (B, M, 3)
    Returns:
        Chamfer Distance (scalar)
    """
    # p1: (B, N, 3), p2: (B, M, 3)
    B, N, _ = p1.shape
    B, M, _ = p2.shape

    p1 = p1.unsqueeze(2)  # (B, N, 1, 3)
    p2 = p2.unsqueeze(1)  # (B, 1, M, 3)

    dist = torch.norm(p1 - p2, dim=3)  # (B, N, M)

    cd_forward = torch.min(dist, dim=2)[0]  # (B, N)
    cd_backward = torch.min(dist, dim=1)[0]  # (B, M)

    return (cd_forward.mean(dim=1) + cd_backward.mean(dim=1)).mean()


def train_model(model, dataloader, epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            pc_t1 = batch['pc_t_minus_1'].to(device)  # (B, N, 3)
            pc_t = batch['pc_t'].to(device)
            pc_t2 = batch['pc_t_plus_1'].to(device)
            gt = batch['gt_dense'].to(device)  # (B, M, 3)

            optimizer.zero_grad()
            pred = model(pc_t1, pc_t, pc_t2)
            loss = chamfer_distance(pred, gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

# Model from previous response
model = TemporalUpsamplingModel(feature_dim=128, up_factor=4)

# Data
dataset = SyntheticPointCloudDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Train
train_model(model, loader, epochs=20, device='cuda' if torch.cuda.is_available() else 'cpu')