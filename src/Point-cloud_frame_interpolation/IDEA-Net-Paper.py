import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEmbedding(nn.Module):
    def __init__(self):
        super(FeatureEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(512, 512, 1)
        self.mlp = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, 3, N)
        x1 = F.relu(self.conv1(x))  # (batch, 64, N)
        x2 = F.relu(self.conv2(x1))  # (batch, 64, N)
        x3 = F.relu(self.conv3(x2))  # (batch, 128, N)
        x4 = F.relu(self.conv4(x3))  # (batch, 256, N)
        x_concat = torch.cat([x1, x2, x3, x4], dim=1)  # (batch, 512, N)
        x5 = F.relu(self.conv5(x_concat))  # (batch, 512, N)
        x6 = torch.max(x5, dim=2, keepdim=True)[0]  # (batch, 512, 1)
        x7 = torch.mean(x5, dim=2, keepdim=True)  # (batch, 512, 1)
        y1 = torch.cat([x6, x7], dim=1).expand(-1, -1, x5.size(2))  # (batch, 1024, N)
        y2_input = torch.cat([x5, y1], dim=1)  # (batch, 1536, N)
        y2 = self.mlp(y2_input.transpose(1, 2))  # (batch, N, 128)
        return y2

class TemporalConsistency(nn.Module):
    def __init__(self, N=1024):
        super(TemporalConsistency, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(N, 1024),
            nn.ReLU(),
            nn.Linear(1024, N)
        )

    def forward(self, F0, F1):
        dist = torch.cdist(F0, F1, p=2) ** 2 + 1e-6  # (batch, N, N)
        A_tilde = 1 / dist
        batch_size = A_tilde.size(0)
        A = self.mlp(A_tilde.view(batch_size * N, N)).view(batch_size, N, N)
        A = F.softmax(A, dim=2)
        return A

class TrajectoryCompensation(nn.Module):
    def __init__(self):
        super(TrajectoryCompensation, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, 1152),
            nn.ReLU(),
            nn.Linear(1152, 576),
            nn.ReLU(),
            nn.Linear(576, 288),
            nn.ReLU(),
            nn.Linear(288, 3),
            nn.Tanh()
        )

    def forward(self, F0_t, F1_t):
        x = torch.cat([F0_t, F1_t], dim=2)  # (batch, N, 256)
        delta = self.mlp(x)  # (batch, N, 3)
        return delta

class IDEA_Net(nn.Module):
    def __init__(self, N=1024):
        super(IDEA_Net, self).__init__()
        self.feature_embedding = FeatureEmbedding()
        self.temporal_consistency = TemporalConsistency(N)
        self.trajectory_compensation = TrajectoryCompensation()

    def forward(self, P0, P1, t):
        F0, F1 = self.feature_embedding(P0, P1)
        A = self.temporal_consistency(F0, F1)
        P0_t = (1 - t) * P0 + t * torch.bmm(A, P1)
        F0_t = (1 - t) * F0 + t * torch.bmm(A, F1)
        P1_t = (1 - t) * torch.bmm(A.transpose(1, 2), P0) + t * P1
        F1_t = (1 - t) * torch.bmm(A.transpose(1, 2), F0) + t * F1
        delta0_t = self.trajectory_compensation(F0_t, F1_t)
        delta1_t = self.trajectory_compensation(F0_t, F1_t)
        O0_t = P0_t + delta0_t
        O1_t = P1_t + delta1_t
        return O0_t, O1_t