"""
    PyTorch implementation of Interaction Networks (IN) and Propagation Networks (PropNet)
    Based on "Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class InteractionNetwork(nn.Module):
    def __init__(self, object_dim, relation_dim, effect_dim, hidden_dim):
        super().__init__()
        self.relation_encoder = MLP(2 * object_dim + relation_dim, hidden_dim, effect_dim)
        self.object_encoder = MLP(object_dim, hidden_dim, effect_dim)
        self.effect_aggregator = MLP(effect_dim + object_dim, hidden_dim, object_dim)

    def forward(self, objects, relations, senders, receivers):
        obj_enc = self.object_encoder(objects)
        rel_inputs = torch.cat([objects[senders], objects[receivers], relations], dim=-1)
        rel_effects = self.relation_encoder(rel_inputs)
        agg_effects = torch.zeros_like(objects)
        agg_effects.index_add_(0, receivers, rel_effects)
        effect_inputs = torch.cat([objects, agg_effects], dim=-1)
        updated_objects = self.effect_aggregator(effect_inputs)
        return updated_objects


class PropagationNetwork(nn.Module):
    def __init__(self, object_dim, relation_dim, effect_dim, hidden_dim, L=3):
        super().__init__()
        self.L = L
        self.relation_encoder = MLP(2 * object_dim + relation_dim, hidden_dim, effect_dim)
        self.object_encoder = MLP(object_dim, hidden_dim, effect_dim)
        self.effect_aggregator = MLP(effect_dim + object_dim * 2, hidden_dim, object_dim)
        self.output_decoder = MLP(object_dim, hidden_dim, object_dim)

    def forward(self, objects, relations, senders, receivers):
        B, N, D = objects.shape
        v_hat = torch.zeros_like(objects)

        obj_enc = self.object_encoder(objects)
        rel_enc = self.relation_encoder(torch.cat([
            objects[:, senders], objects[:, receivers], relations
        ], dim=-1))

        for l in range(self.L):
            rel_inputs = torch.cat([
                rel_enc, v_hat[:, senders], v_hat[:, receivers]
            ], dim=-1)
            e_hat = self.relation_encoder(rel_inputs)

            agg_e = torch.zeros_like(objects)
            agg_e.index_add_(1, receivers, e_hat)

            obj_inputs = torch.cat([obj_enc, agg_e, v_hat], dim=-1)
            v_hat = self.effect_aggregator(obj_inputs)

        return self.output_decoder(v_hat)


# Dataset class for toy particle simulation
class ParticleDataset(Dataset):
    def __init__(self, num_samples=1000, num_particles=5, obj_dim=6, rel_dim=4):
        self.data = []
        for _ in range(num_samples):
            objects = torch.randn(num_particles, obj_dim)
            relations = torch.randn(num_particles * (num_particles - 1), rel_dim)
            senders, receivers = [], []
            for i in range(num_particles):
                for j in range(num_particles):
                    if i != j:
                        senders.append(i)
                        receivers.append(j)
            senders = torch.tensor(senders)
            receivers = torch.tensor(receivers)
            target = torch.randn_like(objects)  # dummy target for supervised learning
            self.data.append((objects, relations, senders, receivers, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Training loop for a generic model (IN or PropNet)
def train_model(model, dataloader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for objects, relations, senders, receivers, target in dataloader:
            objects = objects.unsqueeze(0)  # add batch dimension
            relations = relations.unsqueeze(0)
            optimizer.zero_grad()
            output = model(objects, relations, senders, receivers)
            loss = F.mse_loss(output.squeeze(0), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")


if __name__ == '__main__':
    N = 5
    D = 6
    R = 4
    dataset = ParticleDataset(num_samples=200, num_particles=N, obj_dim=D, rel_dim=R)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    in_model = InteractionNetwork(object_dim=D, relation_dim=R, effect_dim=16, hidden_dim=64)
    optimizer = torch.optim.Adam(in_model.parameters(), lr=1e-3)
    train_model(in_model, dataloader, optimizer)

    propnet = PropagationNetwork(object_dim=D, relation_dim=R, effect_dim=16, hidden_dim=64, L=3)
    optimizer = torch.optim.Adam(propnet.parameters(), lr=1e-3)
    train_model(propnet, dataloader, optimizer)