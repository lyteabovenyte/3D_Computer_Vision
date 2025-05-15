"""
    FlowNet3D is composed of three main blocks:
	1.	Set Conv Layers (Feature Extraction)
	2.	Flow Embedding Layer (Core innovation)
	3.	Set Upconv Layers (Flow Regression), Using interpolation + skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """K-Nearest Neighbors search."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """Extract features for k-nearest neighbors."""
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class SetConv(nn.Module):
    """Set Convolution Layer."""
    def __init__(self, num_points, radius, k, in_channels, out_channels):
        super(SetConv, self).__init__()
        self.num_points = num_points
        self.radius = radius
        self.k = k
        
        layers = []
        for out_channel in out_channels:
            layers.append(nn.Conv2d(in_channels, out_channel, kernel_size=1, bias=True))
            layers.append(nn.BatchNorm2d(out_channel, eps=0.001))
            layers.append(nn.ReLU())
            in_channels = out_channel
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, points, features):
        batch_size = points.size(0)
        # Subsample points
        fps_idx = torch.randperm(points.size(1))[:self.num_points]
        new_points = points[:, fps_idx, :]
        
        # Get k-nearest neighbors
        feature = get_graph_feature(points, k=self.k)  # (batch_size, 3*2, num_points, k)
        feature = self.conv(feature)  # (batch_size, out_channels, num_points, k)
        feature = feature.max(dim=-1, keepdim=False)[0]  # (batch_size, out_channels, num_points)
        
        return new_points, feature

class FlowEmbedding(nn.Module):
    """Flow Embedding Layer."""
    def __init__(self, k, in_channels, out_channels):
        super(FlowEmbedding, self).__init__()
        self.k = k
        
        layers = []
        for out_channel in out_channels:
            layers.append(nn.Conv2d(in_channels, out_channel, kernel_size=1, bias=True))
            layers.append(nn.BatchNorm2d(out_channel, eps=0.001))
            layers.append(nn.ReLU())
            in_channels = out_channel
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, points1, points2, features1, features2):
        batch_size, num_dims, num_points = points2.size()
        
        # Compute k-nearest neighbors from points2 to points1
        points2_t = points2.permute(0, 2, 1).contiguous()  # (batch_size, num_points, 3)
        points1_t = points1.permute(0, 2, 1).contiguous()
        
        dist = torch.cdist(points2_t, points1_t)  # (batch_size, num_points2, num_points1)
        _, ind = dist.topk(self.k, dim=-1, largest=False)  # (batch_size, num_points2, k)
        
        # Interpolate features
        inverse_dist = 1.0 / (dist.gather(2, ind) + 1e-10)
        norm = inverse_dist.sum(dim=2, keepdim=True)
        weights = inverse_dist / norm
        
        idx_flat = ind.view(batch_size, -1)
        features1_flat = features1.permute(0, 2, 1).contiguous().view(batch_size, -1, features1.size(1))
        gathered_features = features1_flat.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, features1.size(1)))
        gathered_features = gathered_features.view(batch_size, num_points, self.k, features1.size(1))
        
        new_features = (gathered_features * weights.unsqueeze(-1)).sum(dim=2)  # (batch_size, num_points, in_channels)
        new_features = torch.cat([new_features.permute(0, 2, 1), features2], dim=1)  # (batch_size, in_channels+features2_channels, num_points)
        
        new_features = new_features.unsqueeze(-1)  # (batch_size, channels, num_points, 1)
        new_features = self.conv(new_features).squeeze(-1)  # (batch_size, out_channels, num_points)
        
        return new_features

class SetUpConv(nn.Module):
    """Set UpConvolution Layer."""
    def __init__(self, k, in_channels1, in_channels2, skip_channels, out_channels):
        super(SetUpConv, self).__init__()
        self.k = k
        
        layers = []
        in_channels = in_channels1 + in_channels2 + sum(skip_channels)
        for out_channel in out_channels:
            layers.append(nn.Conv2d(in_channels, out_channel, kernel_size=1, bias=True))
            layers.append(nn.BatchNorm2d(out_channel, eps=0.001))
            layers.append(nn.ReLU())
            in_channels = out_channel
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, points1, points2, features1, features2, skip_features=None):
        batch_size, num_dims, num_points = points2.size()
        
        # Compute k-nearest neighbors from points2 to points1
        points2_t = points2.permute(0, 2, 1).contiguous()
        points1_t = points1.permute(0, 2, 1).contiguous()
        
        dist = torch.cdist(points2_t, points1_t)
        _, ind = dist.topk(self.k, dim=-1, largest=False)
        
        # Interpolate features
        inverse_dist = 1.0 / (dist.gather(2, ind) + 1e-10)
        norm = inverse_dist.sum(dim=2, keepdim=True)
        weights = inverse_dist / norm
        
        idx_flat = ind.view(batch_size, -1)
        features1_flat = features1.permute(0, 2, 1).contiguous().view(batch_size, -1, features1.size(1))
        gathered_features = features1_flat.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, features1.size(1)))
        gathered_features = gathered_features.view(batch_size, num_points, self.k, features1.size(1))
        
        new_features = (gathered_features * weights.unsqueeze(-1)).sum(dim=2)  # (batch_size, num_points, in_channels)
        new_features = new_features.permute(0, 2, 1)  # (batch_size, in_channels, num_points)
        
        # Concatenate with features2 and skip_features
        new_features = torch.cat([new_features, features2], dim=1)
        if skip_features is not None:
            new_features = torch.cat([new_features, skip_features], dim=1)
        
        new_features = new_features.unsqueeze(-1)
        new_features = self.conv(new_features).squeeze(-1)
        
        return new_features

class FeaturePropagation(nn.Module):
    """Feature Propagation Layer."""
    def __init__(self, in_channels, out_channels, mlp):
        super(FeaturePropagation, self).__init__()
        layers = []
        for out_channel in mlp:
            layers.append(nn.Conv1d(in_channels, out_channel, kernel_size=1, bias=True))
            layers.append(nn.BatchNorm1d(out_channel, eps=0.001))
            layers.append(nn.ReLU())
            in_channels = out_channel
        
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, points1, points2, features1, features2):
        batch_size, num_dims, num_points = points2.size()
        
        # Compute k-nearest neighbors from points2 to points1
        points2_t = points2.permute(0, 2, 1).contiguous()
        points1_t = points1.permute(0, 2, 1).contiguous()
        
        dist = torch.cdist(points2_t, points1_t)
        _, ind = dist.topk(3, dim=-1, largest=False)
        
        # Interpolate features
        inverse_dist = 1.0 / (dist.gather(2, ind) + 1e-10)
        norm = inverse_dist.sum(dim=2, keepdim=True)
        weights = inverse_dist / norm
        
        idx_flat = ind.view(batch_size, -1)
        features1_flat = features1.permute(0, 2, 1).contiguous().view(batch_size, -1, features1.size(1))
        gathered_features = features1_flat.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, features1.size(1)))
        gathered_features = gathered_features.view(batch_size, num_points, 3, features1.size(1))
        
        new_features = (gathered_features * weights.unsqueeze(-1)).sum(dim=2)  # (batch_size, num_points, in_channels)
        new_features = new_features.permute(0, 2, 1)  # (batch_size, in_channels, num_points)
        
        new_features = self.conv(new_features)
        return new_features

class FlowNet3D(nn.Module):
    """FlowNet3D Architecture."""
    def __init__(self):
        super(FlowNet3D, self).__init__()
        
        # Point Feature Learning
        self.set_conv1 = SetConv(1024, 0.5, 16, 3, [32, 32, 64])
        self.set_conv2 = SetConv(256, 1.0, 16, 64, [64, 64, 128])
        
        # Flow Embedding
        self.flow_embedding = FlowEmbedding(64, 128, [128, 128, 128])
        
        # Point Mixture
        self.set_conv3 = SetConv(64, 2.0, 8, 128, [128, 128, 256])
        self.set_conv4 = SetConv(16, 4.0, 8, 256, [256, 256, 512])
        
        # Flow Refinement
        self.set_upconv1 = SetUpConv(8, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(8, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(8, 256, 64, [128, 128, 256], [256])
        
        # Flow Prediction
        self.fp = FeaturePropagation(256, 3, [256, 256])
    
    def forward(self, points1, points2):
        # points1, points2: (batch_size, 3, num_points)
        
        # Point Feature Learning
        points1_1, features1_1 = self.set_conv1(points1, points1)
        points1_2, features1_2 = self.set_conv2(points1_1, features1_1)
        
        points2_1, features2_1 = self.set_conv1(points2, points2)
        points2_2, features2_2 = self.set_conv2(points2_1, features2_1)
        
        # Flow Embedding
        flow_embed = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)
        
        # Point Mixture
        points2_3, features2_3 = self.set_conv3(points2_2, flow_embed)
        points2_4, features2_4 = self.set_conv4(points2_3, features2_3)
        
        # Flow Refinement
        up_features1 = self.set_upconv1(points2_3, points2_4, features2_4, features2_3)
        up_features2 = self.set_upconv2(points2_2, points2_3, up_features1, features2_2, features2_3)
        up_features3 = self.set_upconv3(points2_1, points2_2, up_features2, features2_1, features2_2)
        
        # Flow Prediction
        flow = self.fp(points1, points2_1, up_features3, features2_1)
        
        return flow  # (batch_size, 3, num_points)