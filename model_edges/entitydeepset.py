from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EntityDeepSet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=128, output_dim=5, pool="sum"):
        super().__init__()
        self.pool = pool
        self.phi = MLP(input_dim, hidden_dim // 2, hidden_dim)
        pooled_dim = hidden_dim * 2 if pool == "sum_mean" else hidden_dim
        self.rho = MLP(pooled_dim, hidden_dim, embedding_dim)
        self.head = MLP(embedding_dim, embedding_dim, output_dim) if output_dim is not None else None

    def encode(self, edge_sets):
        pooled = []
        for edge_set in edge_sets:
            h = self.phi(edge_set)
            if self.pool == "sum_mean":
                pooled.append(torch.cat([h.sum(dim=0), h.mean(dim=0)], dim=0))
            else:
                pooled.append(h.sum(dim=0))
        return self.rho(torch.stack(pooled, dim=0))

    def forward(self, edge_sets):
        z = self.encode(edge_sets)
        if self.head is None:
            return z
        return self.head(z)
