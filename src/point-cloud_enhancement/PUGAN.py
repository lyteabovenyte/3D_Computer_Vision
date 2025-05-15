import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    """
        Generator consist of:
        1. Feature extraction
        2. Feature expansion
        3. Coordinate reconstruction
    """
    def __init__(self, up_ratio=4):
        super(Generator, self).__init__()
        self.up_ratio = up_ratio
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU()
        )
        # Feature expansion layers
        self.expansion = nn.Sequential(
            nn.Conv1d(256, 256 * up_ratio, 1),
            nn.ReLU()
        )
        # Coordinate reconstruction layers
        self.coordinate_reconstruction = nn.Sequential(
            nn.Conv1d(256, 3, 1)
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = x.transpose(1, 2)  # [B, 3, N]
        features = self.feature_extractor(x)  # [B, 256, N]
        expanded_features = self.expansion(features)  # [B, 256 * up_ratio, N]
        # Reshape to [B, 256, N * up_ratio]
        B, C, N = features.size()
        expanded_features = expanded_features.view(B, 256, N * self.up_ratio)
        coordinates = self.coordinate_reconstruction(expanded_features)  # [B, 3, N * up_ratio]
        coordinates = coordinates.transpose(1, 2)  # [B, N * up_ratio, 3]
        return coordinates
    

class Discriminator(nn.Module):
    """
        Discriminator consist of:
        1. Feature Extraction
        2. Classification Layers: Fully connected layers to output a probability indicating the authenticity of the input
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # probability of being real
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = x.transpose(1, 2)  # [B, 3, N]
        features = self.feature_extractor(x)  # [B, 256, N]
        # Global feature by max pooling
        global_feature = torch.max(features, 2)[0]  # [B, 256]
        out = self.classifier(global_feature)  # [B, 1]
        return out

# ---------------------------------------
# training loop based on Adversarial loss
# ---------------------------------------
def chamfer_distance(pc1, pc2):
    # Implement Chamfer Distance calculation
    #! This is a placeholder function. You need to implement the actual Chamfer Distance calculation.
    return torch.mean(torch.norm(pc1 - pc2, dim=2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
lambda_rec = 0.1 # Weight for reconstruction loss (Uniformity loss)

# Initialize networks
generator = Generator(up_ratio=4).to(device)
discriminator = Discriminator().to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# Loss functions
bce_loss = nn.BCELoss()

for epoch in range(num_epochs):
    for data in dataloader:
        sparse_pc, dense_pc = data  # [B, N, 3], [B, N * up_ratio, 3]
        sparse_pc = sparse_pc.to(device)
        dense_pc = dense_pc.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        discriminator.zero_grad()
        real_labels = torch.ones(sparse_pc.size(0), 1).to(device)
        fake_labels = torch.zeros(sparse_pc.size(0), 1).to(device)

        outputs = discriminator(dense_pc)
        d_loss_real = bce_loss(outputs, real_labels)

        generated_pc = generator(sparse_pc)
        outputs = discriminator(generated_pc.detach())
        d_loss_fake = bce_loss(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # -----------------
        # Train Generator
        # -----------------
        generator.zero_grad()
        outputs = discriminator(generated_pc)
        g_adv_loss = bce_loss(outputs, real_labels)

        # Compute reconstruction loss (e.g., Chamfer Distance)
        g_rec_loss = chamfer_distance(generated_pc, dense_pc)

        # Total generator loss
        g_loss = g_adv_loss + lambda_rec * g_rec_loss
        g_loss.backward()
        g_optimizer.step()