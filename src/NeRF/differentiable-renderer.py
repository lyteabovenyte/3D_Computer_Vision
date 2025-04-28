"""
    Implicit 3D representation using MLP
    Differentiable rendering using ray sampling and volume rendering
    supervised with posed 2D images via a differentiable renderer
    
    This code is a simplified version of the NeRF model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImplicitMLP(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 3))  # Output RGB
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x is (batch, 3)
        return torch.sigmoid(self.network(x))  # RGB in [0, 1]
    
# differentiable renderer (Ray Sampling)
def generate_rays(H, W, focal, camera_pose):
    """
    Generate rays for a pinhole camera.
    H, W: image height, width
    focal: focal length
    camera_pose: 4x4 pose matrix
    """
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    dirs = torch.stack([(i - W / 2) / focal,
                        -(j - H / 2) / focal,
                        -torch.ones_like(i)], dim=-1)  # [H, W, 3]

    rays_d = torch.sum(dirs[..., None, :] * camera_pose[:3, :3], dim=-1)  # Rotate
    rays_o = camera_pose[:3, 3].expand(rays_d.shape)  # Origin stays same
    return rays_o, rays_d

# Volume Rendering (Ray Integration)
def render_rays(rays_o, rays_d, network, N_samples=64, near=2.0, far=6.0):
    """
    Sample points along the ray and integrate color
    """
    H, W, _ = rays_o.shape
    t_vals = torch.linspace(near, far, N_samples)
    t_vals = t_vals.view(1, 1, N_samples).expand(H, W, N_samples)

    # Points along the rays: x = o + td
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]  # [H, W, N_samples, 3]
    pts_flat = pts.view(-1, 3)

    # Query MLP
    rgb = network(pts_flat).view(H, W, N_samples, 3)

    # Simple alpha compositing (weights = uniform)
    rgb_out = rgb.mean(dim=2)  # Average over samples (for simplicity)
    return rgb_out


# Supervision with 2D images

# Fake "ground truth" RGB image (just for demo)
H, W = 64, 64
target_image = torch.rand(H, W, 3)

# Camera setup
focal = 50.0
camera_pose = torch.eye(4)

# Initialize model and optimizer
model = ImplicitMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for step in range(500):
    rays_o, rays_d = generate_rays(H, W, focal, camera_pose)
    rgb_pred = render_rays(rays_o, rays_d, model)
    loss = F.mse_loss(rgb_pred, target_image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")