import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D U-Net Implementation
# It outputs an initial segmentation map with out_channels classes.
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        # Encoder (Contracting Path)
        self.enc1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        
        # Decoder (Expansive Path)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2) # for the skip connections
        self.dec2 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # for the skip connections
        self.dec1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        
        # Final layer
        self.final = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        
        # Bottleneck
        b = F.relu(self.bottleneck(self.pool(e2)))
        
        # Decoder with skip connections
        d2 = self.upconv2(b)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = F.relu(self.dec2(d2))
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = F.relu(self.dec1(d1))
        
        return self.final(d1)

# Relation Network Implementation

class RelationNetwork(nn.Module):
    def __init__(self, feature_dim, relation_dim):
        """
        feature_dim: dimension of the feature vector for each point (from U-Net)
        relation_dim: dimension of the relation score
        """
        super(RelationNetwork, self).__init__()
        self.relation_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1)
        )

    def forward(self, features):
        # Features: (batch_size, num_points, feature_dim)
        batch_size, num_points, feature_dim = features.size()
        
        # Compute pairwise relations
        relations = []
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    pair = torch.cat([features[:, i, :], features[:, j, :]], dim=-1)
                    relation_score = self.relation_mlp(pair)
                    relations.append(relation_score)
        
        relations = torch.stack(relations, dim=1)  # (batch_size, num_pairs, 1)
        return relations

# Combined Model
class UNetRelationNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dim, relation_dim):
        super(UNetRelationNet, self).__init__()
        self.unet = UNet3D(in_channels, out_channels)
        self.feature_extractor = nn.Conv3d(64, feature_dim, kernel_size=1)  # Extract features from U-Net
        self.relation_net = RelationNetwork(feature_dim, relation_dim)
        self.final_conv = nn.Conv3d(out_channels + 1, out_channels, kernel_size=1)

    def forward(self, x):
        # Step 1: U-Net for initial segmentation
        unet_out = self.unet(x)  # (batch_size, out_channels, D, H, W)
        
        # Step 2: Extract features from U-Net's decoder
        features = self.feature_extractor(unet_out)  # (batch_size, feature_dim, D, H, W)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  # (batch_size, num_points, feature_dim)
        
        # Step 3: Relation Network to refine
        relations = self.relation_net(features)  # (batch_size, num_pairs, 1)
        relations = relations.view(unet_out.size(0), 1, unet_out.size(2), unet_out.size(3), unet_out.size(4))
        
        # Step 4: Combine U-Net output and Relation Network output
        combined = torch.cat([unet_out, relations], dim=1)
        refined_out = self.final_conv(combined)
        
        return refined_out