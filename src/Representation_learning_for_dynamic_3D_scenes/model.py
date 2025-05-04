import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSceneRepresentation(nn.Module):
    def __init__(self, latent_dim=128, num_frames=16, feature_dim=64):
        super(DynamicSceneRepresentation, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        
        # Spatial-temporal encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, feature_dim, kernel_size=(3,3,3), padding=1),
            nn.ReLU()
        )
        
        # Temporal dynamics encoder
        self.temporal_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Decoder for scene reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 3, kernel_size=(3,3,3), padding=1),
            nn.Sigmoid()
        )
        
        # Coordinate MLP for implicit representation
        self.coord_mlp = nn.Sequential(
            nn.Linear(3 + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def encode(self, scene_sequence):
        """
        scene_sequence: (batch, frames, channels, depth, height, width)
        """
        batch_size = scene_sequence.size(0)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(scene_sequence)
        
        # Reshape for temporal encoding
        spatial_features = spatial_features.permute(0, 2, 1, 3, 4)  # (batch, frames, channels, depth, height)
        spatial_features = spatial_features.reshape(
            batch_size, self.num_frames, -1
        )
        
        # Temporal encoding
        latent, (h_n, c_n) = self.temporal_encoder(spatial_features)
        
        return latent, h_n

    def decode(self, latent, coords=None):
        """
        latent: (batch, frames, latent_dim)
        coords: optional (batch, num_points, 3) for implicit representation
        """
        if coords is not None:
            # Implicit representation
            batch_size, num_points, _ = coords.size()
            latent_expanded = latent.unsqueeze(2).expand(-1, -1, num_points, -1)
            coords_expanded = coords.unsqueeze(1).expand(-1, latent.size(1), -1, -1)
            
            mlp_input = torch.cat([coords_expanded, latent_expanded], dim=-1)
            mlp_input = mlp_input.reshape(-1, 3 + self.latent_dim)
            
            output = self.coord_mlp(mlp_input)
            output = output.reshape(batch_size, latent.size(1), num_points, 3)
            
            return output
        else:
            # Explicit reconstruction
            latent = latent.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            recon = self.decoder(latent)
            return recon

    def forward(self, scene_sequence, coords=None):
        latent, _ = self.encode(scene_sequence)
        output = self.decode(latent, coords)
        return output, latent


def train_step(model, scene_sequence, coords=None, optimizer=None):
    model.train()
    optimizer.zero_grad()
    
    recon, latent = model(scene_sequence, coords)
    
    # Reconstruction loss
    if coords is None:
        recon_loss = F.mse_loss(recon, scene_sequence)
    else:
        target = scene_sequence[:, :, :, :3]  # Assuming RGB target
        recon_loss = F.mse_loss(recon, target)
    
    # Add regularization on latent
    latent_loss = 0.01 * torch.mean(latent ** 2)
    
    loss = recon_loss + latent_loss
    loss.backward()
    optimizer.step()
    
    return loss.item()
