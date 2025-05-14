"""
    Semantic Point Cloud Upsampling based on the paper: https://arxiv.org/abs/2012.04439
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class PointClassificationNet(nn.Module):
    def __init__(self, input_dim=3, num_classes=40):
        super(PointClassificationNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.cls_head = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.mlp(x)  # [B, N, 256]
        logits = self.cls_head(features.mean(dim=1))  # [B, num_classes]
        return logits, features  # return logits and semantic features


class GACFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super(GACFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, N]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # [B, out_channels, N]
        x = x.transpose(1, 2)  # [B, N, out_channels]
        return x


class KeyPositionAttention(nn.Module):
    def __init__(self, feature_dim=256):
        super(KeyPositionAttention, self).__init__()
        self.key_proj = nn.Linear(feature_dim, 64)
        self.query_proj = nn.Linear(feature_dim, 64)
        self.value_proj = nn.Linear(feature_dim, 256)

    def forward(self, features):
        # Self-attention style
        Q = self.query_proj(features)
        K = self.key_proj(features)
        V = self.value_proj(features)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # [B, N, N]
        attention = F.softmax(scores, dim=-1)
        attended = torch.bmm(attention, V)
        return attended  # [B, N, 256]


class EnhancedUpsamplingModule(nn.Module):
    def __init__(self, in_dim=256, up_factor=2):
        super(EnhancedUpsamplingModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * up_factor)  # Output XYZ coordinates
        )
        self.up_factor = up_factor

    def forward(self, features):
        B, N, C = features.shape
        upsampled = self.mlp(features)  # [B, N, 3*up_factor]
        upsampled = upsampled.view(B, N * self.up_factor, 3)  # [B, N*up, 3]
        return upsampled


class SPUModel(nn.Module):
    def __init__(self, num_classes=40, up_factor=2):
        super(SPUModel, self).__init__()
        self.point_cls_net = PointClassificationNet(input_dim=3, num_classes=num_classes)
        self.feature_extractor = GACFeatureExtractor()
        self.key_attention = KeyPositionAttention()
        self.upsampling_module = EnhancedUpsamplingModule(up_factor=up_factor)
        self.classifier = PointClassificationNet(input_dim=3, num_classes=num_classes)

    def forward(self, sparse_pc):
        # 1. Semantic prior extraction
        _, semantic_features = self.point_cls_net(sparse_pc)

        # 2. GAC-based feature extraction
        local_features = self.feature_extractor(sparse_pc)

        # 3. Multiply with attention
        attention_features = self.key_attention(local_features)
        fused = local_features + attention_features

        # 4. Enhanced Upsampling Module
        upsampled_pc = self.upsampling_module(fused)

        # 5. Classification for supervision
        classification_logits, _ = self.classifier(upsampled_pc)

        return upsampled_pc, classification_logits
"""

class PointClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=50):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.cls_head = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, N, 3]
        feat = self.mlp(x)               # [B, N, 256]
        global_feat = feat.mean(dim=1)   # [B, 256]
        logits = self.cls_head(global_feat)
        return logits, feat


class GACExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, out_channels, 1)
        )

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(1, 2)             # [B, C, N]
        x = self.conv(x)
        return x.transpose(1, 2)          # [B, N, out_channels]


class KeyPositionAttention(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.q_proj = nn.Linear(dim, 64)
        self.k_proj = nn.Linear(dim, 64)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)
        return out


class PreInterpolation(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class EnhancedUpsampleModule(nn.Module):
    def __init__(self, in_dim=256, up_factor=2):
        super().__init__()
        self.up_factor = up_factor
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * up_factor)
        )

    def forward(self, x):
        B, N, _ = x.shape
        up = self.mlp(x)                   # [B, N, 3*up]
        return up.view(B, N * self.up_factor, 3)


class SPU(nn.Module):
    def __init__(self, num_classes=50, up_factor=2, stages=2):
        super().__init__()
        self.stages = stages
        self.up_factor = up_factor

        self.classifier = PointClassifier(3, num_classes)
        self.feature_extractors = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.interpolators = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for _ in range(stages):
            self.feature_extractors.append(GACExtractor(3, 256))
            self.attentions.append(KeyPositionAttention(256))
            self.interpolators.append(PreInterpolation(256))
            self.upsamplers.append(EnhancedUpsampleModule(256, up_factor))

        self.final_cls = PointClassifier(3, num_classes)

    def forward(self, pc):
        # pc: [B, N, 3]
        cls_logits, _ = self.classifier(pc)
        current_pc = pc

        outputs = []
        for i in range(self.stages):
            f = self.feature_extractors[i](current_pc)
            a = self.attentions[i](f)
            f_att = f + a

            f_interp = self.interpolators[i](f_att)
            current_pc = self.upsamplers[i](f_interp)
            outputs.append(current_pc)

        final_logits, _ = self.final_cls(current_pc)
        return current_pc, final_logits, outputs
    
def training_step(model, input_pc, gt_labels, optimizer, cls_loss_fn):
    model.train()
    upsampled_pc, logits, _ = model(input_pc)
    
    cls_loss = cls_loss_fn(logits, gt_labels)
    
    optimizer.zero_grad()
    cls_loss.backward()
    optimizer.step()
    
    return cls_loss.item()