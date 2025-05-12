import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def knn(x, k):
    # x: (B, C, N), returns idx: (B, N, k)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    # x: (B, C, N), return edge features: (B, 2C, N, k)
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base  # (B, N, k)
    idx = idx.view(-1)  # (B * N * k)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
    return feature


class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, num_classes=40):
        super(DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (B, C, N)

        x1 = get_graph_feature(x, k=self.k)  # (B, 2C, N, k)
        x1 = self.conv1(x1).max(dim=-1)[0]  # (B, 64, N)

        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2).max(dim=-1)[0]  # (B, 64, N)

        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3).max(dim=-1)[0]  # (B, 128, N)

        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4).max(dim=-1)[0]  # (B, 256, N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        x = self.conv5(x_cat)  # (B, emb_dims, N)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), dim=1)

        x = self.dp1(F.leaky_relu(self.bn6(self.linear1(x)), 0.2))
        x = self.dp2(F.leaky_relu(self.bn7(self.linear2(x)), 0.2))
        x = self.linear3(x)
        return x