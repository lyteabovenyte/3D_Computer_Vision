"""
    Example Bilateral upsampling which is used in the last stage of progressive upsampling method.

    bilateral weights computation:
    $$w_{ij} = \exp\left(-\frac{||x_i - x_j||^2}{\sigma_x^2} - \frac{||f_i - f_j||^2}{\sigma_f^2}\right)$$
"""
import torch

# Helper knn function
def knn(xyz1, xyz2, k):
    """
    xyz1: [B, M, 3] - query points (fine)
    xyz2: [B, N, 3] - key points (coarse)
    Return: idx [B, M, k] - indices of k nearest neighbors in xyz2 for each point in xyz1
    """
    B, M, _ = xyz1.shape
    _, N, _ = xyz2.shape
    dists = torch.cdist(xyz1, xyz2)  # [B, M, N]
    _, idx = dists.topk(k, largest=False, dim=-1)  # [B, M, k]
    return idx

def bilateral_interpolation(coarse_xyz, coarse_feat, fine_xyz, fine_feat=None, k=3, sigma_x=0.1, sigma_f=1.0):
    """
    Inputs:
        coarse_xyz: [B, N, 3] - points from previous layer
        coarse_feat: [B, N, C] - features from previous layer
        fine_xyz: [B, M, 3] - points at current layer
        fine_feat: [B, M, C] - (optional) features at current layer (for feature distance)
    Returns:
        interpolated_feat: [B, M, C]
    """
    B, N, C = coarse_feat.shape
    _, M, _ = fine_xyz.shape

    if fine_feat is None:
        fine_feat = torch.zeros(B, M, C, device=coarse_feat.device)

    # Find k-NN from fine points to coarse points
    idx = knn(fine_xyz, coarse_xyz, k)  # [B, M, k]

    # Gather coarse_xyz and coarse_feat
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    neighbor_xyz = torch.gather(coarse_xyz.unsqueeze(1).expand(-1, M, -1, -1), 2, idx_expand)  # [B, M, k, 3]
    
    idx_expand_feat = idx.unsqueeze(-1).expand(-1, -1, -1, C)
    neighbor_feat = torch.gather(coarse_feat.unsqueeze(1).expand(-1, M, -1, -1), 2, idx_expand_feat)  # [B, M, k, C]

    # Compute spatial distance
    fine_xyz_exp = fine_xyz.unsqueeze(2).expand(-1, -1, k, -1)  # [B, M, k, 3]
    d_x = torch.norm(fine_xyz_exp - neighbor_xyz, dim=-1)  # [B, M, k]

    # Compute feature distance
    fine_feat_exp = fine_feat.unsqueeze(2).expand(-1, -1, k, -1)
    d_f = torch.norm(fine_feat_exp - neighbor_feat, dim=-1)  # [B, M, k]

    # Compute bilateral weights
    w = torch.exp(- (d_x ** 2) / sigma_x**2 - (d_f ** 2) / sigma_f**2)  # [B, M, k]
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize

    # Interpolate features
    interpolated_feat = torch.sum(w.unsqueeze(-1) * neighbor_feat, dim=2)  # [B, M, C]

    return interpolated_feat


B, N, M, C = 2, 512, 1024, 64
coarse_xyz = torch.rand(B, N, 3)
coarse_feat = torch.rand(B, N, C)
fine_xyz = torch.rand(B, M, 3)
fine_feat = torch.rand(B, M, C)  # optional

output = bilateral_interpolation(coarse_xyz, coarse_feat, fine_xyz, fine_feat, k=3)
print(output.shape)  # [B, M, C]