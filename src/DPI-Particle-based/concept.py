"""
    Consider the context below:

    We leverage dynamic particle interaction networks for control tasks
    in both simulation and the real world. Because trajectory optimisation
    using the shooting method can easily be stuck to a local minimum, 
    we first randomly sample N-sample control sequences, similar
    to the MPPI algorithm, and select the
    best-performing one according to the rollouts of our learned model.
    We then optimise it via the shooting method using our model's gradients. 
    We also use online system identification to further improve
    the model's performance.

    below is a realistic implementation of the control loop using a learned
    particle-based dynamic model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParticleDynamicsModel(nn.Module):
    """
    A simplified but realistic version of an Interaction Network for particle dynamics.
    contains message passing and update particles based on their relations.
    The model predicts the next state of particles based on their current state
    and the relations with other particles.
    """
    def __init__(self, particle_dim, relation_dim, hidden_dim):
        super().__init__()
        self.relation_encoder = nn.Sequential(
            nn.Linear(particle_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, relation_dim),
            nn.ReLU(),
        )

        self.dynamics_fn = nn.Sequential(
            nn.Linear(particle_dim + relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, particle_dim),
        )

    def forward(self, particles, adjacency_matrix):
        B, N, D = particles.shape
        relations = []
        for i in range(N):
            for j in range(N):
                if adjacency_matrix[i, j] == 1:
                    sender = particles[:, i, :]  # [B, D]
                    receiver = particles[:, j, :]  # [B, D]
                    edge_input = torch.cat([sender, receiver], dim=-1)
                    relation_feat = self.relation_encoder(edge_input) # relation_dim
                    relations.append((j, relation_feat))

        # Aggregate relation features per particle
        relation_tensor = torch.zeros(B, N, relations[0][1].shape[-1], device=particles.device)
        for j, feat in relations:
            relation_tensor[:, j, :] += feat  # aggregate messages

        # Concatenate and predict next state
        dynamics_input = torch.cat([particles, relation_tensor], dim=-1)
        delta = self.dynamics_fn(dynamics_input)  # predict velocity delta or position delta
        return particles + delta  # simple Euler integration


class ParticleController:
    """
    Control policy using sampled trajectory optimization and gradient-based shooting.
    """
    def __init__(self, model, horizon=10, n_samples=100, action_dim=2, device='cpu'):
        self.model = model
        self.horizon = horizon
        self.n_samples = n_samples
        self.action_dim = action_dim
        self.device = device

    def sample_action_sequences(self, B):
        """Random action sequences for MPPI-style initialization."""
        return torch.randn(B, self.horizon, self.action_dim, device=self.device) * 0.1

    def rollout(self, particles, action_seq, adjacency_matrix):
        """Rollout dynamics model using a sequence of actions."""
        traj = [particles]
        state = particles
        for t in range(action_seq.shape[1]):
            state = state + action_seq[:, t].unsqueeze(1)  # broadcast same control to all particles
            state = self.model(state, adjacency_matrix)
            traj.append(state)
        return torch.stack(traj, dim=1)  # [B, T+1, N, D]

    def evaluate_trajectories(self, traj, target):
        """Simple loss: final distance to goal."""
        final_state = traj[:, -1]  # [B, N, D]
        return ((final_state - target) ** 2).sum(dim=[1, 2])

    def optimize(self, particles, target, adjacency_matrix):
        """Trajectory optimization using MPPI + shooting."""
        B = self.n_samples
        action_seqs = self.sample_action_sequences(B).detach().requires_grad_(True)

        # Initial rollout + evaluation
        with torch.no_grad():
            trajs = self.rollout(particles.repeat(B, 1, 1), action_seqs, adjacency_matrix)
            losses = self.evaluate_trajectories(trajs, target.repeat(B, 1, 1))
            best_idx = torch.argmin(losses)

        # Refine best with gradient-based shooting
        best_action_seq = action_seqs[best_idx:best_idx + 1].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([best_action_seq], lr=0.05)

        for _ in range(10):
            optimizer.zero_grad()
            traj = self.rollout(particles, best_action_seq, adjacency_matrix)
            loss = self.evaluate_trajectories(traj, target).mean()
            loss.backward()
            optimizer.step()

        return best_action_seq.detach()


# Hook for online system identification (placeholder)
def online_update(model, real_data):
    """
    Adapt model parameters using new real-world data (e.g., via gradient descent).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for i in range(5):
        pred = model(real_data['state'], real_data['adj'])
        loss = ((pred - real_data['next_state']) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
