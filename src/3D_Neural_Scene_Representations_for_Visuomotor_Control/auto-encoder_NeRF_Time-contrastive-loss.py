"""
    example code for embedding neural radiance fields into and autoencoder and
    time contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple encoder (e.g., from image to latent)
class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Simplified NeRF decoder (RGB + density)
class NeRFDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 + 3 + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # RGB (3) + Density (1)
        )

    def forward(self, x, d, z):
        # x, d: (B, 3), z: (B, latent_dim)
        input = torch.cat([x, d, z], dim=-1) # adding latent code to input
        return self.mlp(input)  # (B, 4)

# InfoNCE-style Time-Contrastive Loss
def time_contrastive_loss(z_all, temperature=0.07):
    """
    z_all: (T, D) latent embeddings at different time steps
    """
    T, D = z_all.shape
    z_all = F.normalize(z_all, dim=-1)  # normalize for cosine similarity

    sim_matrix = torch.matmul(z_all, z_all.T) / temperature  # (T, T)
    labels = torch.arange(T - 1)

    # Only compare z[t] and z[t+1] (positive pairs)
    logits = sim_matrix[:-1]  # shape: (T-1, T)
    pos = torch.diag(sim_matrix, diagonal=1)  # positive similarities

    # Mask out the diagonal (self-similarity)
    mask = ~torch.eye(T, dtype=torch.bool)
    logits = logits.masked_select(mask[:-1]).view(T - 1, T - 1)

    # Apply cross entropy
    return F.cross_entropy(logits, labels)


def main():
    batch_size = 5  # number of time steps
    latent_dim = 32
    img_size = 64

    encoder = ImageEncoder(latent_dim)
    decoder = NeRFDecoder(latent_dim)

    images = torch.randn(batch_size, 3, img_size, img_size)  # (T, C, H, W)

    # encode
    z_all = encoder(images)  # (T, latent_dim)

    # time contrastive loss
    tcl = time_contrastive_loss(z_all)

    # NeRF decoder rendering (simulate one ray sample)
    coords = torch.randn(batch_size, 3)
    dirs = torch.randn(batch_size, 3)
    outputs = decoder(coords, dirs, z_all)  # (T, 4)

    print(f"TCL loss: {tcl.item():.4f}")
    print(f"Decoder output shape: {outputs.shape}")

if __name__ == "__main__":
    main()