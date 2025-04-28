import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 1. The NeRF MLP (Neural Radiance Field)
# Nerf predicts RGB color and density for a given 3D point and viewing direction
# --------------------------
class RealNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_dir=27):
        """
        D: number of layers for density encoder
        W: number of hidden units
        input_ch: number of input channels for position (after positional encoding)
        input_ch_dir: number of input channels for direction (after positional encoding)
        """
        super(RealNeRF, self).__init__()

        self.D = D
        self.W = W

        # Positional Encoding Input for (x,y,z)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W + input_ch, W) if i == 4 else nn.Linear(W, W) for i in range(1, D)] # skip connection at every 4 layer
        )

        # Output for density
        self.sigma_linear = nn.Linear(W, 1)

        # Feature vector for RGB branch
        self.feature_linear = nn.Linear(W, W)

        # RGB branch
        self.rgb_linear_1 = nn.Linear(W + input_ch_dir, W // 2)
        self.rgb_linear_2 = nn.Linear(W // 2, 3)

    def forward(self, x, d):
        """
        x: (N, input_ch) position inputs (after positional encoding)
        d: (N, input_ch_dir) view directions (after positional encoding)
        """
        h = x
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i == 4:
                h = torch.cat([x, h], -1)

        sigma = F.relu(self.sigma_linear(h))  # Density output

        feature = self.feature_linear(h)

        h = torch.cat([feature, d], -1)
        h = F.relu(self.rgb_linear_1(h))
        rgb = torch.sigmoid(self.rgb_linear_2(h))  # Color output

        return rgb, sigma
    
# Sinusoids applied to inputs
# This is a positional encoding function that maps input coordinates to a higher-dimensional space
# using sine and cosine functions. This is crucial for NeRF to learn high-frequency details.
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)

    def forward(self, x):
        """
        x: (..., input_dims)
        returns: (..., input_dims * (2*num_freqs) [+ input_dims if include_input])
        """
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, -1)

# --------------------------
# 2. Volume Rendering
# --------------------------
def volume_rendering(rgb, sigma, z_vals, dirs):
    """
    rgb: (N_samples, 3) colors at each sample
    sigma: (N_samples, 1) densities at each sample
    z_vals: (N_samples,) distances along the ray
    dirs: (3,) ray direction
    """
    deltas = z_vals[1:] - z_vals[:-1]   # Distance between adjacent samples
    deltas = torch.cat([deltas, torch.tensor([1e10]).to(deltas.device)])  # Infinity for the last one

    # Compute alpha by converting density and distance into probability of stopping
    alpha = 1.0 - torch.exp(-sigma.squeeze() * deltas)  # (N_samples,)

    # Compute accumulated transmittance T
    T = torch.cumprod(torch.cat([torch.ones(1).to(alpha.device), 1. - alpha + 1e-10])[:-1], dim=0)

    # Weight for each sample
    weights = T * alpha  # (N_samples,)

    # Rendered color is weighted sum of colors
    rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=0)  # (3,)
    
    # Optional: depth map can also be computed
    depth_map = (weights * z_vals).sum()

    return rgb_map, depth_map