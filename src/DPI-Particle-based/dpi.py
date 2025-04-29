"""
    PyTorch implementation of Interaction Networks (IN) and Propagation Networks (PropNet)
    Based on "Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_scatter


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

# based on https://arxiv.org/pdf/1809.11169
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


class HierarchicalDynamicsModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Local interaction model (particle to particle)
        self.local_interaction = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Particle to cluster aggregation
        self.p_to_c_agg = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        # Cluster to cluster interaction
        self.cluster_interaction = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Cluster to particle broadcast
        self.c_to_p_broadcast = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        # Final update model
        self.final_update = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU()
        )

    def forward(self, particle_states, cluster_states, particle_to_cluster, particle_edges, cluster_edges):
        """
        particle_states: (N, D)
        cluster_states: (C, D)
        particle_to_cluster: (N,) -> cluster ID for each particle
        particle_edges: (E_p, 2) -> [sender, receiver] pairs for particle-level graph
        cluster_edges: (E_c, 2) -> [sender, receiver] pairs for cluster-level graph
        """

        # ---- Particle-to-particle interaction ----
        sender_states = particle_states[particle_edges[:, 0]]
        receiver_states = particle_states[particle_edges[:, 1]]
        edge_input = torch.cat([sender_states, receiver_states], dim=-1)
        delta_particles_local = self.local_interaction(edge_input)

        # Aggregate messages to each particle
        particle_msg = torch.zeros_like(particle_states)
        particle_msg = particle_msg.index_add(0, particle_edges[:, 1], delta_particles_local)

        # ---- Particle-to-cluster aggregation (bottom-up) ----
        cluster_inputs = self.p_to_c_agg(particle_states)
        cluster_states_agg = torch_scatter.scatter_mean(cluster_inputs, particle_to_cluster, dim=0)

        # ---- Cluster-to-cluster interaction ----
        sender_c = cluster_states_agg[cluster_edges[:, 0]]
        receiver_c = cluster_states_agg[cluster_edges[:, 1]]
        c_edge_input = torch.cat([sender_c, receiver_c], dim=-1)
        delta_clusters = self.cluster_interaction(c_edge_input)
        cluster_msg = torch.zeros_like(cluster_states_agg)
        cluster_msg = cluster_msg.index_add(0, cluster_edges[:, 1], delta_clusters)

        # ---- Cluster-to-particle broadcast (top-down) ----
        cluster_msg_broadcast = cluster_msg[particle_to_cluster]
        cluster_msg_transformed = self.c_to_p_broadcast(cluster_msg_broadcast)

        # ---- Combine local and global messages ----
        combined_msg = torch.cat([particle_msg, cluster_msg_transformed], dim=-1)
        updated_particle_states = self.final_update(combined_msg)

        return updated_particle_states