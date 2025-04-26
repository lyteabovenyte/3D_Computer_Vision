import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(TransitionModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], state_dim * 2)  # Predict mean and log_std

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        return mean, log_std

class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 2)  # Predict mean and log_std

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        return mean, log_std

class TerminationModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(TerminationModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)  # Predict termination probability

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
import numpy as np

class MPPIPlanner:
    def __init__(self, dynamics_model, action_dim, horizon, num_samples, lambda_):
        self.dynamics_model = dynamics_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_

    def plan(self, current_state):
        # Sample action sequences
        action_sequences = np.random.randn(self.num_samples, self.horizon, self.action_dim)
        returns = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            state = current_state
            total_reward = 0
            for t in range(self.horizon):
                action = action_sequences[i, t]
                mean, log_std = self.dynamics_model.transition_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                                                                      torch.tensor(action, dtype=torch.float32).unsqueeze(0))
                next_state = mean.detach().numpy().squeeze()
                reward_mean, _ = self.dynamics_model.reward_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                                                                  torch.tensor(action, dtype=torch.float32).unsqueeze(0))
                reward = reward_mean.detach().numpy().squeeze()
                total_reward += reward
                state = next_state
            returns[i] = total_reward

        # Compute weights
        max_return = np.max(returns)
        weights = np.exp((returns - max_return) / self.lambda_)
        weights /= np.sum(weights)

        # Compute weighted average of action sequences
        optimal_sequence = np.tensordot(weights, action_sequences, axes=(0, 0))
        return optimal_sequence[0]  # Return the first action in the sequence