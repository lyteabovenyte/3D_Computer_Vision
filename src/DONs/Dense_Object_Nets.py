"""
    Source paper: Keypoints into the Future: Self-Supervised Correspondence in Model-Based Reinforcement Learning
    https://arxiv.org/pdf/2009.05085
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List

# Configuration
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Input image size (as per typical RL tasks)
NUM_KEYPOINTS = 10  # Number of keypoints to detect
ACTION_DIM = 4  # Action space (e.g., delta x, y, z, theta for robotic arm)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

class KeypointExtractor(nn.Module):
    """CNN to extract keypoints from input images."""
    def __init__(self, num_keypoints: int):
        super(KeypointExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute feature map size
        self.feature_size = IMG_HEIGHT // 8  # Assuming stride=2 reduces size by 2^3
        self.keypoint_head = nn.Conv2d(512, num_keypoints, kernel_size=1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x (B, 3, H, W) - Batch of RGB images
        Output: keypoint_probs (B, NUM_KEYPOINTS, H', W') - Keypoint probability maps
        """
        features = self.conv_layers(x)
        keypoint_probs = self.keypoint_head(features)
        return keypoint_probs

class DenseCorrespondenceModel(nn.Module):
    """Model for self-supervised dense correspondence learning."""
    def __init__(self, num_keypoints: int):
        super(DenseCorrespondenceModel, self).__init__()
        self.keypoint_extractor = KeypointExtractor(num_keypoints)
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.correspondence_head = nn.Conv2d(128, num_keypoints, kernel_size=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: img1, img2 (B, 3, H, W) - Pair of images for correspondence
        Output: correspondence_map (B, NUM_KEYPOINTS, H', W'), keypoint_probs (B, NUM_KEYPOINTS, H', W')
        """
        # Extract keypoints from first image
        keypoint_probs = self.keypoint_extractor(img1)
        # Encode features for correspondence
        features = self.keypoint_extractor.conv_layers(img2)
        encoded_features = self.feature_encoder(features)
        correspondence_map = self.correspondence_head(encoded_features)
        return correspondence_map, keypoint_probs

class DynamicsModel(nn.Module):
    """Model to predict future keypoint positions given current keypoints and action."""
    def __init__(self, num_keypoints: int, action_dim: int):
        super(DynamicsModel, self).__init__()
        self.keypoint_dim = 2  # (x, y) coordinates
        self.mlp = nn.Sequential(
            nn.Linear(num_keypoints * self.keypoint_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_keypoints * self.keypoint_dim),
        )

    def forward(self, keypoints: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Input: 
            keypoints (B, NUM_KEYPOINTS * 2) - Current keypoint coordinates
            action (B, ACTION_DIM) - Action taken
        Output: next_keypoints (B, NUM_KEYPOINTS * 2) - Predicted next keypoint coordinates
        """
        x = torch.cat([keypoints, action], dim=-1)
        next_keypoints = self.mlp(x)
        return next_keypoints

def sample_keypoints(prob_maps: torch.Tensor, num_keypoints: int) -> torch.Tensor:
    """
    Sample keypoint coordinates from probability maps.
    Input: prob_maps (B, NUM_KEYPOINTS, H', W')
    Output: keypoints (B, NUM_KEYPOINTS, 2) - (x, y) coordinates
    """
    B, K, H, W = prob_maps.shape
    prob_maps = prob_maps.view(B, K, -1)
    prob_maps = F.softmax(prob_maps, dim=-1)
    indices = torch.multinomial(prob_maps, 1).squeeze(-1)  # (B, K)
    y = indices // W
    x = indices % W
    keypoints = torch.stack([x, y], dim=-1).float()  # (B, K, 2)
    return keypoints

def correspondence_loss(correspondence_map: torch.Tensor, keypoint_probs: torch.Tensor, 
                       target_keypoints: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for dense correspondence pretext task.
    Uses cross-entropy loss between predicted correspondence map and target keypoint locations.
    """
    B, K, H, W = correspondence_map.shape
    target_indices = (target_keypoints[:, :, 1] * W + target_keypoints[:, :, 0]).long()
    correspondence_map = correspondence_map.view(B, K, -1)
    loss = F.cross_entropy(correspondence_map, target_indices)
    return loss

def dynamics_loss(pred_keypoints: torch.Tensor, target_keypoints: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss for keypoint dynamics prediction."""
    return F.mse_loss(pred_keypoints, target_keypoints)

def train_step(correspondence_model: DenseCorrespondenceModel, 
               dynamics_model: DynamicsModel,
               img1: torch.Tensor, img2: torch.Tensor, 
               action: torch.Tensor, target_keypoints: torch.Tensor,
               optimizer: optim.Optimizer) -> Tuple[float, float]:
    """Single training step for correspondence and dynamics models."""
    correspondence_model.train()
    dynamics_model.train()
    optimizer.zero_grad()

    # Correspondence learning
    correspondence_map, keypoint_probs = correspondence_model(img1, img2)
    sampled_keypoints = sample_keypoints(keypoint_probs, NUM_KEYPOINTS)
    corr_loss = correspondence_loss(correspondence_map, keypoint_probs, sampled_keypoints)

    # Dynamics prediction
    pred_keypoints = dynamics_model(sampled_keypoints.view(BATCH_SIZE, -1), action)
    dyn_loss = dynamics_loss(pred_keypoints, target_keypoints.view(BATCH_SIZE, -1))

    # Combined loss
    total_loss = corr_loss + dyn_loss
    total_loss.backward()
    optimizer.step()
    return corr_loss.item(), dyn_loss.item()

def main():
    # Initialize models
    correspondence_model = DenseCorrespondenceModel(NUM_KEYPOINTS)
    dynamics_model = DynamicsModel(NUM_KEYPOINTS, ACTION_DIM)
    optimizer = optim.Adam(
        list(correspondence_model.parameters()) + list(dynamics_model.parameters()),
        lr=LEARNING_RATE
    )

    # Dummy data for demonstration (replace with real dataset)
    img1 = torch.randn(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH)
    img2 = torch.randn(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH)
    action = torch.randn(BATCH_SIZE, ACTION_DIM)
    target_keypoints = torch.randn(BATCH_SIZE, NUM_KEYPOINTS, 2)

    # Training loop
    for epoch in range(100):
        corr_loss, dyn_loss = train_step(
            correspondence_model, dynamics_model, 
            img1, img2, action, target_keypoints, optimizer
        )
        print(f"Epoch {epoch}, Corr Loss: {corr_loss:.4f}, Dyn Loss: {dyn_loss:.4f}")

if __name__ == "__main__":
    main()