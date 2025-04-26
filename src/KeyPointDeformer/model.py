import torch
import torch.nn as nn
import torch.nn.functional as F

# PointNet-based KeypointDeformer
class KeypointDeformer(nn.Module):
    def __init__(self, num_keypoints=8, point_dim=3, latent_dim=64, cage_size=4):
        super(KeypointDeformer, self).__init__()
        self.num_keypoints = num_keypoints
        self.cage_size = cage_size
        
        # PointNet-like Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(point_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Keypoint Predictor
        self.keypoint_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_keypoints * 3)  # Outputs x, y, z for each keypoint
        )
        
        # Cage Deformation Network
        self.cage_deform_net = nn.Sequential(
            nn.Linear(num_keypoints * 3, 128),
            nn.ReLU(),
            nn.Linear(128, cage_size * cage_size * cage_size * 3)  # Cage vertex displacements
        )
    
    def forward(self, source_points, target_points):
        # source_points, target_points: (batch_size, num_points, 3)
        
        # Encode source and target point clouds
        source_latent = self.encoder(source_points.permute(0, 2, 1)).max(dim=2)[0]  # (batch_size, latent_dim)
        target_latent = self.encoder(target_points.permute(0, 2, 1)).max(dim=2)[0]
        
        # Predict keypoints
        source_keypoints = self.keypoint_predictor(source_latent).view(-1, self.num_keypoints, 3)
        target_keypoints = self.keypoint_predictor(target_latent).view(-1, self.num_keypoints, 3)
        
        # Apply Farthest Point Sampling (FPS) regularization 
        source_keypoints = self.fps_regularization(source_points, source_keypoints)
        target_keypoints = self.fps_regularization(target_points, target_keypoints)
        
        # Compute keypoint differences
        keypoint_diff = target_keypoints - source_keypoints  # (batch_size, num_keypoints, 3)
        
        # Predict cage deformation
        cage_displacement = self.cage_deform_net(keypoint_diff.view(-1, self.num_keypoints * 3))
        cage_displacement = cage_displacement.view(-1, self.cage_size, self.cage_size, self.cage_size, 3)
        
        # Apply cage deformation to source points (simplified interpolation)
        deformed_points = self.apply_cage_deformation(source_points, cage_displacement)
        
        return deformed_points, source_keypoints, target_keypoints
    
    def fps_regularization(self, points, keypoints):
        # Simplified Farthest Point Sampling to distribute keypoints
        # Initialize with predicted keypoints
        batch_size, num_points, _ = points.size()
        updated_keypoints = keypoints.clone()
        
        for b in range(batch_size):
            # Compute distances from points to keypoints
            dists = torch.cdist(points[b], keypoints[b])  # (num_points, num_keypoints)
            min_dists, _ = dists.min(dim=1)  # Distance to nearest keypoint
            # Select point farthest from existing keypoints
            farthest_idx = min_dists.argmax()
            updated_keypoints[b, 0] = points[b, farthest_idx]
            
            # Update remaining keypoints iteratively 
            for i in range(1, self.num_keypoints):
                dists = torch.cdist(points[b], updated_keypoints[b, :i])
                min_dists, _ = dists.min(dim=1)
                farthest_idx = min_dists.argmax()
                updated_keypoints[b, i] = points[b, farthest_idx]
        
        return updated_keypoints
    
    def apply_cage_deformation(self, points, cage_displacement):
        # Simplified cage-based deformation
        # Assume a 3D grid (cage) of size cage_size^3
        batch_size, num_points, _ = points.size()
        cage_size = self.cage_size
        
        # Create a normalized cage grid (0 to 1)
        grid = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, cage_size),
            torch.linspace(0, 1, cage_size),
            torch.linspace(0, 1, cage_size)
        ), dim=-1).to(points.device)  # (cage_size, cage_size, cage_size, 3)
        
        # Apply displacement to cage
        deformed_cage = grid + cage_displacement
        
        # Interpolate points within the deformed cage (simplified trilinear)
        deformed_points = points.clone()
        for b in range(batch_size):
            for i in range(num_points):
                # Normalize point to [0,1] for cage coordinates
                p = (points[b, i] - points[b].min(dim=0)[0]) / (
                    points[b].max(dim=0)[0] - points[b].min(dim=0)[0] + 1e-6)
                
                # Find the cage cell containing the point
                idx = (p * (cage_size - 1)).long().clamp(0, cage_size - 2)
                u, v, w = idx[0], idx[1], idx[2]
                
                # Simplified trilinear interpolation
                weights = p * (cage_size - 1) - idx.float()
                wx, wy, wz = weights[0], weights[1], weights[2]
                
                # Interpolate cage displacements
                d = (1 - wx) * (1 - wy) * (1 - wz) * deformed_cage[b, u, v, w] + \
                    wx * (1 - wy) * (1 - wz) * deformed_cage[b, u + 1, v, w] + \
                    (1 - wx) * wy * (1 - wz) * deformed_cage[b, u, v + 1, w] + \
                    wx * wy * (1 - wz) * deformed_cage[b, u + 1, v + 1, w] + \
                    (1 - wx) * (1 - wy) * wz * deformed_cage[b, u, v, w + 1] + \
                    wx * (1 - wy) * wz * deformed_cage[b, u + 1, v, w + 1] + \
                    (1 - wx) * wy * wz * deformed_cage[b, u, v + 1, w + 1] + \
                    wx * wy * wz * deformed_cage[b, u + 1, v + 1, w + 1]
                
                deformed_points[b, i] = points[b, i] + d
        
        return deformed_points

# Chamfer Distance Loss
def chamfer_distance(pred_points, target_points):
    batch_size, num_points, _ = pred_points.size()
    pred_points = pred_points.view(batch_size, num_points, 1, 3)
    target_points = target_points.view(batch_size, 1, num_points, 3)
    distances = torch.sum((pred_points - target_points) ** 2, dim=-1)
    loss1 = torch.mean(distances.min(dim=2)[0])
    loss2 = torch.mean(distances.min(dim=1)[0])
    return loss1 + loss2

# Training Step
def train_step(model, source_points, target_points, optimizer):
    model.train()
    optimizer.zero_grad()
    
    deformed_points, source_keypoints, target_keypoints = model(source_points, target_points)
    loss = chamfer_distance(deformed_points, target_points)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example Usage
if __name__ == "__main__":
    # Dummy point clouds (batch_size=2, num_points=1024, dim=3)
    source_points = torch.rand(2, 1024, 3)
    target_points = torch.rand(2, 1024, 3)
    
    # Initialize model and optimizer
    model = KeypointDeformer(num_keypoints=8, cage_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run a training step
    loss = train_step(model, source_points, target_points, optimizer)
    print(f"Training loss: {loss}")