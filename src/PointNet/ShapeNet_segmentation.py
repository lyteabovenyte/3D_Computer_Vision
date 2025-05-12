"""
    applying PointNet to ShapeNet segmentation task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import PointNetEncoder, feature_transform_regularizer


# dataset Loader
def load_h5_data_label_seg(h5_filename):
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]          # (N, 2048, 3)
        label = f['label'][:]        # (N, 1)
        seg = f['pid'][:]            # (N, 2048)
    return data, label, seg

class ShapeNetPartDataset(Dataset):
    def __init__(self, root_dir, split='train', npoints=2048, normalize=True):
        self.npoints = npoints
        self.normalize = normalize
        self.data = []
        self.labels = []
        self.segs = []

        # List of HDF5 files for the specified split
        split_files = {
            'train': 'train_data_files.txt',
            'val': 'val_data_files.txt',
            'test': 'test_data_files.txt'
        }

        with open(os.path.join(root_dir, split_files[split]), 'r') as f:
            file_list = [line.strip() for line in f]

        for h5_filename in file_list:
            data, label, seg = load_h5_data_label_seg(os.path.join(root_dir, h5_filename))
            self.data.append(data)
            self.labels.append(label)
            self.segs.append(seg)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.segs = np.concatenate(self.segs, axis=0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        point_set = self.data[idx][:self.npoints]
        label = self.labels[idx]
        seg = self.segs[idx][:self.npoints]

        if self.normalize:
            point_set = point_set - np.mean(point_set, axis=0)
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
            point_set = point_set / dist

        point_set = torch.from_numpy(point_set.astype(np.float32)).transpose(0, 1)  # (3, N)
        label = torch.from_numpy(label.astype(np.int64))
        seg = torch.from_numpy(seg.astype(np.int64))

        return point_set, label, seg
    

# Initialize dataset
train_dataset = ShapeNetPartDataset(root_dir='path_to_data', split='train')

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)


class PointNetPartSeg(nn.Module):
    def __init__(self, num_part=50, num_classes=16, feature_transform=False):
        super(PointNetPartSeg, self).__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform

        self.feat = PointNetEncoder(global_feat=False, feature_transform=feature_transform)

        self.conv1 = nn.Conv1d(1088 + num_classes, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, num_part, 1)  # per-point predictions

    def forward(self, x, label):
        B, D, N = x.size()
        # x: (B, 3, N), label: (B, 1)
        x, trans, trans_feat = self.feat(x)  # x: (B, 1088, N)

        # Expand one-hot category label
        label = label.view(B, 1, 1)
        label_onehot = torch.zeros(B, self.num_classes, device=x.device)
        label_onehot.scatter_(1, label.squeeze(2).long(), 1)
        label_onehot = label_onehot.view(B, self.num_classes, 1).repeat(1, 1, N)

        x = torch.cat([x, label_onehot], dim=1)  # (B, 1088 + num_classes, N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # (B, num_part, N)

        return x, trans, trans_feat
    

model = PointNetPartSeg(num_part=50, num_classes=16, feature_transform=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    for points, class_labels, seg_labels in train_loader:
        points = points.transpose(2, 1).to(device)  # (B, 3, N)
        class_labels = class_labels.to(device)      # (B, 1)
        seg_labels = seg_labels.to(device)          # (B, N)

        pred, trans, trans_feat = model(points, class_labels)
        pred = pred.permute(0, 2, 1).contiguous()   # (B, N, num_part)

        loss = criterion(pred.view(-1, 50), seg_labels.view(-1))
        loss += feature_transform_regularizer(trans_feat) * 0.001

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()