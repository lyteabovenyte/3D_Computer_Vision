import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)

    def forward(self, x):
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(out, dim=-1)
    
class DeformationField(nn.Module):
    def __init__(self, pos_dim=3, time_dim=1, hidden_dim=128, pe_freqs=10, time_pe_freqs=4):
        super().__init__()
        self.pe = PositionalEncoding(pe_freqs)
        self.time_pe = PositionalEncoding(time_pe_freqs)

        input_dim = pe_freqs * 2 * pos_dim + time_pe_freqs * 2 * time_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # 3D offset

    def forward(self, x, t):
        x_pe = self.pe(x)
        t_pe = self.time_pe(t)
        input = torch.cat([x_pe, t_pe], dim=-1)

        h = F.relu(self.fc1(input))
        h = F.relu(self.fc2(h))
        delta_x = self.fc3(h)  # predicted offset

        return x + delta_x  # dynamic point warped to canonical space
    

class CanonicalNeRF(nn.Module):
    def __init__(self, pos_dim=3, dir_dim=3, hidden_dim=256, pe_freqs=10, dir_pe_freqs=4):
        super().__init__()
        self.pe = PositionalEncoding(pe_freqs)
        self.dir_pe = PositionalEncoding(dir_pe_freqs)

        input_ch = pe_freqs * 2 * pos_dim
        input_dir = dir_pe_freqs * 2 * dir_dim

        self.fc1 = nn.Linear(input_ch, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1 + hidden_dim)  # density + features

        self.rgb_layer1 = nn.Linear(hidden_dim + input_dir, hidden_dim // 2)
        self.rgb_layer2 = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x, d):
        x_pe = self.pe(x)
        d_pe = self.dir_pe(d)

        h = F.relu(self.fc1(x_pe))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        sigma_feat = self.fc4(h)

        sigma = F.relu(sigma_feat[..., 0])
        features = sigma_feat[..., 1:]

        h = torch.cat([features, d_pe], dim=-1)
        h = F.relu(self.rgb_layer1(h))
        rgb = torch.sigmoid(self.rgb_layer2(h))  # values in [0, 1]

        return rgb, sigma
    

class DNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.deformation = DeformationField()
        self.canonical_nerf = CanonicalNeRF()

    def forward(self, x, d, t):
        # x: [B, 3], d: [B, 3], t: [B, 1]
        x_canonical = self.deformation(x, t)
        rgb, sigma = self.canonical_nerf(x_canonical, d)
        return rgb, sigma