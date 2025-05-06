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

# supporting multi-view inputs and applying hard negative pairs on contrastive-loss.
def time_contrastive_loss(z_all, temperature=0.07, hard_negative_window=2):
    """
    Args:
        z_all: (T, V, D) tensor of embeddings
        temperature: scalar for scaling logits
        hard_negative_window: how many steps away from t to treat as hard negatives
    Returns:
        contrastive loss scalar
    """
    T, V, D = z_all.shape
    device = z_all.device
    z_all = F.normalize(z_all, dim=-1)  # Normalize for cosine similarity

    # Flatten time and view for anchor-positive setup
    z_flat = z_all.view(T * V, D)  # (T*V, D)

    # Create positive pairs: same time, different view
    anchors = []
    positives = []
    for t in range(T):
        for v in range(V):
            for v_pos in range(V):
                if v != v_pos:
                    anchors.append(z_all[t, v])
                    positives.append(z_all[t, v_pos])
    anchors = torch.stack(anchors)  # (N_pos, D)
    positives = torch.stack(positives)  # (N_pos, D)

    # Compute similarity scores between anchors and all possible targets
    logits = torch.matmul(anchors, z_flat.T) / temperature  # (N_pos, T*V)

    # Create positive labels: index of correct positive in the flattened z_all
    positive_indices = []
    for t in range(T):
        for v in range(V):
            for v_pos in range(V):
                if v != v_pos:
                    idx = t * V + v_pos
                    positive_indices.append(idx)
    labels = torch.tensor(positive_indices, device=device, dtype=torch.long)  # (N_pos,)

    # Optionally mask hard negatives: those at nearby time steps
    if hard_negative_window > 0:
        N_pos = anchors.shape[0]
        mask = torch.ones_like(logits, dtype=torch.bool)
        for i, idx in enumerate(positive_indices):
            t_pos = idx // V
            for dt in range(-hard_negative_window, hard_negative_window + 1):
                t_neighbor = t_pos + dt
                if 0 <= t_neighbor < T:
                    for v in range(V):
                        hard_idx = t_neighbor * V + v
                        mask[i, hard_idx] = False
        logits = logits.masked_fill(~mask, float('-inf'))

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