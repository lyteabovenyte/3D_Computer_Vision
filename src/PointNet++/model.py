import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Sequential(
                nn.Conv2d(last_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            ))
            last_channel = out_channel

    def forward(self, xyz, points):
        #! Sample and group operations would go here, often implemented with a custom CUDA kernel (based on FPS+ball-Query)
        # Assume new_xyz and grouped_points are obtained after sampling and grouping
        new_xyz = None  # Placeholder
        grouped_points = None  # Placeholder

        # Apply MLP to grouped points
        for conv in self.mlp_convs:
            grouped_points = conv(grouped_points)

        # Pooling operation (e.g., max pooling)
        new_points = torch.max(grouped_points, 3)[0]

        return new_xyz, new_points

class PointNet2SSG(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SSG, self).__init__()
        # 3 layer set abstraction
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes) # local features

    def forward(self, xyz):
        B, _, _ = xyz.shape
        points = None
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
    
# continued with feature propagation (upsampling, interpolation, skip-connection(this one is introduced in paper))