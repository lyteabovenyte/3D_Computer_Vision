"""
    self-supervised visual-correspondence learning as an input for a 3D-keypoint-based predictive dynamics model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fueature extractor like a small ResNet (FCN)
class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),  # (B,32,H/2,W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # (B,64,H/4,W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, 3, stride=2, padding=1),  # (B,128,H/8,W/8)
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv_net(x)  # Output feature map
    

# corresponder matcher
class CorrespondenceLearner(nn.Module):
    def __init__(self, feature_dim=128):
        super(CorrespondenceLearner, self).__init__()
        self.feature_extractor = FeatureExtractor(feature_dim)
    
    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)  # (B,128,H/8,W/8)
        feat2 = self.feature_extractor(img2)
        
        # Normalize features
        feat1_norm = F.normalize(feat1, dim=1)
        feat2_norm = F.normalize(feat2, dim=1)

        # Compute dense correlation (dot product between all patches)
        corr = torch.einsum('bchw,bcij->bhwij', feat1_norm, feat2_norm)
        # corr: (batch, H1, W1, H2, W2) — full correspondence map

        return corr
    

class KeypointPredictor(nn.Module):
    def __init__(self, num_keypoints=32, feature_dim=128):
        super(KeypointPredictor, self).__init__()
        self.num_keypoints = num_keypoints
        
        self.conv_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * num_keypoints, 1)  # 3 coordinates per keypoint (x, y, z)
        )
    
    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        kp_pred = self.conv_head(feat_map)  # (B, 3*K, H, W)
        
        # Spatial pooling: average over spatial locations
        kp_pred = kp_pred.mean(dim=[2,3])  # (B, 3*K)
        
        kp_pred = kp_pred.view(B, self.num_keypoints, 3)  # (B, K, 3)
        
        return kp_pred
    

class DynamicsPredictor(nn.Module):
    def __init__(self, num_keypoints=32):
        super(DynamicsPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_keypoints * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_keypoints * 3)
        )
    
    def forward(self, keypoints):
        B, K, _ = keypoints.shape
        flat = keypoints.view(B, -1)  # (B, 3*K)
        out = self.model(flat)
        out = out.view(B, K, 3)
        return out # next position of keypoints in time t+1
    

# Pixel-wise Contrastive Loss
class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_negatives=100):
        super(PixelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def forward(self, feat1, feat2, pos_coords):
        """
        feat1: (B, C, H, W) - feature map from image 1
        feat2: (B, C, H, W) - feature map from image 2
        pos_coords: list of positive pixel coordinates [( (h1, w1), (h2, w2) ), ...] per batch item
        """
        B, C, H, W = feat1.shape
        total_loss = 0.0

        for i in range(B):
            (h1, w1), (h2, w2) = pos_coords[i]

            q = F.normalize(feat1[i, :, h1, w1], dim=0)  # (C,)
            k_pos = F.normalize(feat2[i, :, h2, w2], dim=0)  # (C,)

            # Sample random negative pixels from feat2
            rand_h = torch.randint(0, H, (self.num_negatives,))
            rand_w = torch.randint(0, W, (self.num_negatives,))
            negs = feat2[i, :, rand_h, rand_w]  # (C, num_negatives)
            negs = F.normalize(negs, dim=0)  # normalize each negative feature

            # Compute positive similarity
            sim_pos = torch.matmul(q, k_pos) / self.temperature  # scalar

            # Compute negative similarities
            sim_negs = torch.matmul(q.unsqueeze(0), negs) / self.temperature  # (1, num_negatives)

            logits = torch.cat([sim_pos.unsqueeze(0), sim_negs.squeeze(0)], dim=0)  # (1+num_negatives,)

            labels = torch.zeros(1, dtype=torch.long, device=logits.device)  # positive at position 0

            total_loss += F.cross_entropy(logits.unsqueeze(0), labels)  # Cross-entropy expects batch dim

        return total_loss / B
    

class KeypointLoss(nn.Module):
    def __init__(self):
        super(KeypointLoss, self).__init__()
    
    def forward(self, pred_keypoints, gt_keypoints):
        """
        pred_keypoints: (B, K, 3)
        gt_keypoints: (B, K, 3)
        """
        return F.mse_loss(pred_keypoints, gt_keypoints)
    

class DynamicsLoss(nn.Module):
    def __init__(self):
        super(DynamicsLoss, self).__init__()

    def forward(self, predicted_future_kp, true_future_kp):
        return F.mse_loss(predicted_future_kp, true_future_kp)
    
# ------- Main Training Loop -------
# Instantiate networks
corr_model = CorrespondenceLearner()
kp_model = KeypointPredictor()
dyn_model = DynamicsPredictor()

# Losses
corr_loss_fn = PixelContrastiveLoss()
kp_loss_fn = KeypointLoss()
dyn_loss_fn = DynamicsLoss()

# Optimizers (can be combined too)
optimizer = torch.optim.Adam(list(corr_model.parameters()) +
                              list(kp_model.parameters()) +
                              list(dyn_model.parameters()), lr=1e-4)


for epoch in range(10):
    for batch_idx in range(100):
        img1 = torch.randn(4, 3, 128, 128)
        img2 = torch.randn(4, 3, 128, 128)
        
        # Ground-truth positive pixel coordinates (for now randomly simulate)
        pos_coords = [((8, 8), (8, 8)) for _ in range(4)]  # perfect match at center

        # Ground-truth keypoints (simulate)
        gt_kp_now = torch.randn(4, 32, 3)
        gt_kp_future = gt_kp_now + 0.01 * torch.randn(4, 32, 3)  # small movement

        ## Forward
        feat1 = corr_model.feature_extractor(img1)
        feat2 = corr_model.feature_extractor(img2)
        
        correspondences = corr_model(img1, img2)
        pred_keypoints = kp_model(feat1)
        pred_future_keypoints = dyn_model(pred_keypoints)

        ## Losses
        loss_corr = corr_loss_fn(feat1, feat2, pos_coords)
        loss_kp = kp_loss_fn(pred_keypoints, gt_kp_now)
        loss_dyn = dyn_loss_fn(pred_future_keypoints, gt_kp_future)

        total_loss = loss_corr + loss_kp + loss_dyn

        ## Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Total Loss: {total_loss.item():.4f}")