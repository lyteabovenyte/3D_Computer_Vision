"""
    The architecture integrates a convolutional encoder, a NeRF-based rendering module, 
    a dynamics model, and an auto-decoding mechanism for test-time optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified Convolutional Encoder
class ConvEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(ConvEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)  # Adjust based on input size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flattening x
        return self.fc(x)

# Simplified NeRF Module (Mixture Density Network)
# which returns latents that are used in Dynamic model such as density and color
class NeRFModule(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super(NeRFModule, self).__init__()
        self.pos_enc_dim = 60  # Positional encoding for 3D points (L=10)
        self.input_dim = latent_dim + self.pos_enc_dim + 3  # Latent code + pos + view dir -> input to NeRF decoder
        self.sigma_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Density
        )
        self.color_net = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # RGB color
        )

    def positional_encoding(self, x, L=10):
        """Positional encoding for 3D coordinates"""
        out = []
        for i in range(L):
            out.append(torch.sin(2.0 ** i * x))
            out.append(torch.cos(2.0 ** i * x))
        return torch.cat(out, dim=-1)

    def forward(self, z, points, view_dirs):
        """z: latent code, points: 3D coordinates, view_dirs: viewing directions"""
        points_enc = self.positional_encoding(points, L=10)
        inputs = torch.cat([z, points_enc, view_dirs], dim=-1)
        sigma = self.sigma_net(inputs)
        color_input = torch.cat([inputs, sigma], dim=-1)
        color = self.color_net(color_input)
        return sigma, color

# Dynamics Model
class DynamicsModel(nn.Module):
    """Predicts next latent state given current latent state and action"""
    def __init__(self, latent_dim=128, action_dim=4, hidden_dim=256):
        super(DynamicsModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z, action):
        inputs = torch.cat([z, action], dim=-1)
        return self.net(inputs)

# Time Contrastive Loss
def time_contrastive_loss(z_t, z_t_pos, z_t_neg, temperature=0.07):
    """
    Time contrastive loss for viewpoint-invariant representations
    Args:
        z_t: current latent state
        z_t_pos: positive sample (next state)
        z_t_neg: negative sample (random state)
        temperature: scaling factor for similarity
    Returns:
        loss: computed contrastive loss
    """
    batch_size = z_t.shape[0]
    z_t = F.normalize(z_t, dim=1)
    z_t_pos = F.normalize(z_t_pos, dim=1)
    z_t_neg = F.normalize(z_t_neg, dim=1)

    pos_sim = F.cosine_similarity(z_t, z_t_pos, dim=-1) / temperature
    neg_sim = F.cosine_similarity(z_t, z_t_neg, dim=-1) / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long).to(z_t.device)
    return F.cross_entropy(logits, labels)

# Main Model
class NeuralSceneVisuomotor(nn.Module):
    def __init__(self, latent_dim=128, action_dim=4):
        super(NeuralSceneVisuomotor, self).__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim) # Perception Module
        self.nerf = NeRFModule(latent_dim=latent_dim) # Rendering Module
        self.dynamics = DynamicsModel(latent_dim=latent_dim, action_dim=action_dim) # Dynamics Module

    def forward(self, image, points, view_dirs, action):
        # Encode image to latent code
        z = self.encoder(image)
        # Render using NeRF
        sigma, color = self.nerf(z, points, view_dirs)
        # Predict next latent state
        z_next = self.dynamics(z, action)
        return sigma, color, z, z_next

# Training Loop (Simplified)
def train_step(model, optimizer, images, points, view_dirs, actions, z_pos, z_neg):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    sigma, color, z, z_next = model(images, points, view_dirs, actions)

    # Losses
    # Reconstruction loss (simplified, assumes ground-truth RGB and density)
    recon_loss = F.mse_loss(color, torch.ones_like(color))  # Placeholder
    # Time contrastive loss
    tcl_loss = time_contrastive_loss(z, z_pos, z_neg)
    # Dynamics prediction loss (simplified)
    dynamics_loss = F.mse_loss(z_next, torch.zeros_like(z_next))  # Placeholder

    total_loss = recon_loss + tcl_loss + dynamics_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

# Example usage
if __name__ == "__main__":
    # Dummy data
    batch_size, C, H, W = 8, 3, 64, 64
    latent_dim, action_dim = 128, 4
    images = torch.randn(batch_size, C, H, W)
    points = torch.randn(batch_size, 100, 3)  # 100 sampled 3D points
    view_dirs = torch.randn(batch_size, 100, 3)
    actions = torch.randn(batch_size, action_dim)
    z_pos = torch.randn(batch_size, latent_dim)  # Positive samples
    z_neg = torch.randn(batch_size, latent_dim)  # Negative samples

    # Initialize model and optimizer
    model = NeuralSceneVisuomotor(latent_dim=latent_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for one step
    loss = train_step(model, optimizer, images, points, view_dirs, actions, z_pos, z_neg)
    print(f"Training loss: {loss:.4f}")