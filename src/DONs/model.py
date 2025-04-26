import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Keypoint Encoder ---------
class KeypointEncoder(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointEncoder, self).__init__()
        self.num_keypoints = num_keypoints
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), # RGB input
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, num_keypoints, 1) # Heatmap output
        )
    
    def soft_argmax(self, heatmaps):
        N, K, H, W = heatmaps.shape
        heatmaps = heatmaps.view(N, K, -1)
        heatmaps = F.softmax(heatmaps, dim=-1)  # Softmax over spatial dimensions
        heatmaps = heatmaps.view(N, K, H, W)

        grids_y = torch.linspace(0, 1, steps=H, device=heatmaps.device)
        grids_x = torch.linspace(0, 1, steps=W, device=heatmaps.device)
        gy, gx = torch.meshgrid(grids_y, grids_x, indexing='ij')

        gx = gx.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        gy = gy.unsqueeze(0).unsqueeze(0)

        x = torch.sum(heatmaps * gx, dim=[2, 3])  # (N, K)
        y = torch.sum(heatmaps * gy, dim=[2, 3])  # (N, K)

        keypoints = torch.stack([x, y], dim=-1)  # (N, K, 2)
        return keypoints
    
    def forward(self, img):
        heatmaps = self.conv(img)  # (N, K, H, W)
        keypoints = self.soft_argmax(heatmaps)
        return keypoints

# --------- Keypoint Dynamics Model ---------
class KeypointDynamicsModel(nn.Module):
    def __init__(self, num_keypoints, action_dim):
        super(KeypointDynamicsModel, self).__init__()
        input_dim = num_keypoints * 2 + action_dim
        output_dim = num_keypoints * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, keypoints, action):
        B, K, _ = keypoints.shape
        x = torch.cat([keypoints.view(B, -1), action], dim=-1)
        next_keypoints = self.mlp(x)
        next_keypoints = next_keypoints.view(B, K, 2)
        return next_keypoints

# --------- Planning Module (MPC) ---------
def plan_action(keypoint_model, dynamics_model, current_img, goal_img, action_space, planning_horizon=5, num_candidates=100):
    # Extract keypoints
    current_kp = keypoint_model(current_img)  # (1, K, 2)
    goal_kp = keypoint_model(goal_img)        # (1, K, 2)

    best_action = None
    best_score = float('inf')

    for _ in range(num_candidates):
        # Randomly sample a sequence of actions
        actions = torch.randn((planning_horizon, action_space), device=current_img.device) * 0.1
        
        # Simulate forward
        kp = current_kp.clone()
        for action in actions:
            action = action.unsqueeze(0)  # (1, action_space)
            kp = dynamics_model(kp, action)

        # Compute final distance to goal
        score = F.mse_loss(kp, goal_kp)
        
        if score.item() < best_score:
            best_score = score.item()
            best_action = actions[0]  # Choose first action of best sequence

    return best_action