from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

try:
    from .entitydeepset import MLP
except ImportError:
    from entitydeepset import MLP


def _to_padded(edge_sets):
    if isinstance(edge_sets, tuple):
        return edge_sets
    lengths = torch.tensor([len(edge_set) for edge_set in edge_sets], device=edge_sets[0].device)
    return pad_sequence(edge_sets, batch_first=True), lengths


def _lengths_to_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def _masked_max(x, valid, dim):
    return x.masked_fill(~valid.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=dim).values


def _masked_mean(x, valid, dim):
    valid_f = valid.to(x.dtype)
    return (x * valid_f.unsqueeze(-1)).sum(dim=dim) / valid_f.sum(dim=dim).clamp_min(1.0).unsqueeze(-1)


def _masked_pool(x, valid, pool):
    if pool == "max":
        return _masked_max(x, valid, dim=1)
    if pool == "mean":
        return _masked_mean(x, valid, dim=1)
    if pool == "max_mean":
        return torch.cat([_masked_max(x, valid, dim=1), _masked_mean(x, valid, dim=1)], dim=-1)
    raise ValueError(f"Unknown pool={pool!r}")


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, valid):
        valid = valid.bool()
        flat = x.reshape(-1, x.shape[-1])
        flat_valid = valid.reshape(-1)
        if self.training and flat_valid.any():
            selected = flat[flat_valid]
            mean = selected.mean(dim=0)
            var = selected.var(dim=0, unbiased=False)
            self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.weight + self.bias


class MaskedLinearBNReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = MaskedBatchNorm1d(output_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, valid):
        return self.act(self.bn(self.linear(x), valid))


class MaskedSharedMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(
            [MaskedLinearBNReLU(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x, valid):
        for layer in self.layers:
            x = layer(x, valid)
        return x.masked_fill(~valid.unsqueeze(-1), 0.0)


class TransformNet2D(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        global_dim = max(hidden_dim * 4, 256)
        self.mlp = MaskedSharedMLP([input_dim, 64, hidden_dim, global_dim])
        self.fc = nn.Sequential(
            nn.Linear(global_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim * input_dim),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x, valid):
        pooled = _masked_max(self.mlp(x, valid), valid, dim=1)
        transform = self.fc(pooled).view(-1, self.input_dim, self.input_dim)
        identity = torch.eye(self.input_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        return transform + identity


class EntityPointNet(nn.Module):
    """2D PointNet over ArcSet elements.

    The first two channels are treated as 2D coordinates. All channels remain
    point attributes after the learned 2D input transform.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        embedding_dim=128,
        output_dim=5,
        pool="max",
        dropout=0.0,
        input_transform="2d",
    ):
        super().__init__()
        self.pool = pool
        self.input_transform_mode = input_transform
        if input_transform == "2d":
            self.input_transform = TransformNet2D(2, hidden_dim=hidden_dim)
        elif input_transform == "full":
            self.input_transform = TransformNet2D(input_dim, hidden_dim=hidden_dim)
        elif input_transform == "none":
            self.input_transform = None
        else:
            raise ValueError(f"Unknown input_transform={input_transform!r}; "
                             "use '2d' (PointNet default), 'full' (project all dims), "
                             "or 'none' (drop the T-Net).")
        self.mlp1 = MaskedSharedMLP([input_dim, 64])
        self.feature_transform = TransformNet2D(64, hidden_dim=hidden_dim)
        global_dim = max(hidden_dim * 4, embedding_dim)
        self.mlp2 = MaskedSharedMLP([64, hidden_dim, global_dim])
        pooled_dim = global_dim * 2 if pool == "max_mean" else global_dim
        self.rho = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = MLP(embedding_dim, embedding_dim, output_dim) if output_dim is not None else None

    def encode(self, edge_sets):
        x, lengths = _to_padded(edge_sets)
        valid = _lengths_to_mask(lengths, x.shape[1])

        if self.input_transform_mode == "2d" and self.input_transform is not None:
            coords = x[..., :2]
            coord_transform = self.input_transform(coords, valid)
            coords = torch.bmm(coords, coord_transform)
            x = torch.cat([coords, x[..., 2:]], dim=-1)
        elif self.input_transform_mode == "full" and self.input_transform is not None:
            transform = self.input_transform(x, valid)
            x = torch.bmm(x, transform)
        # else: input_transform_mode == "none" -> pass through

        h = self.mlp1(x, valid)
        feature_transform = self.feature_transform(h, valid)
        h = torch.bmm(h, feature_transform)
        h = self.mlp2(h, valid)
        return self.rho(_masked_pool(h, valid, self.pool))

    def forward(self, edge_sets):
        z = self.encode(edge_sets)
        if self.head is None:
            return z
        return self.head(z)


def _knn_indices(query_coords, point_coords, point_valid, k):
    k = min(k, point_coords.shape[1])
    dist = torch.cdist(query_coords, point_coords)
    dist = dist.masked_fill(~point_valid[:, None, :], torch.finfo(dist.dtype).max)
    return dist.topk(k=k, dim=-1, largest=False).indices


def _gather_points(values, idx):
    b, s, k = idx.shape
    c = values.shape[-1]
    expanded = values[:, None, :, :].expand(b, s, values.shape[1], c)
    return torch.gather(expanded, 2, idx.unsqueeze(-1).expand(b, s, k, c))


def _gather_mask(valid, idx):
    b, s, k = idx.shape
    expanded = valid[:, None, :].expand(b, s, valid.shape[1])
    return torch.gather(expanded, 2, idx)


def _fps_indices(coords, valid, npoint):
    b, n, _ = coords.shape
    npoint = min(npoint, n)
    idx = torch.zeros(b, npoint, dtype=torch.long, device=coords.device)
    valid_counts = valid.sum(dim=1)
    first = valid.to(torch.float32).argmax(dim=1)
    farthest = first
    distances = torch.full((b, n), torch.finfo(coords.dtype).max, device=coords.device)
    batch = torch.arange(b, device=coords.device)

    for i in range(npoint):
        idx[:, i] = farthest
        centroid = coords[batch, farthest].unsqueeze(1)
        dist = ((coords - centroid) ** 2).sum(dim=-1)
        distances = torch.minimum(distances, dist)
        distances = distances.masked_fill(~valid, -1.0)
        farthest = distances.max(dim=1).indices

    center_valid = torch.arange(npoint, device=coords.device)[None, :] < valid_counts[:, None].clamp_max(npoint)
    return idx, center_valid


class SetAbstraction2D(nn.Module):
    def __init__(self, input_dim, output_dim, npoint=32, k=16):
        super().__init__()
        mid_dim = max(output_dim // 2, 32)
        self.npoint = npoint
        self.k = k
        self.local_mlp = MaskedSharedMLP([input_dim + 2, mid_dim, mid_dim, output_dim])

    def forward(self, coords, features, valid):
        center_idx, center_valid = _fps_indices(coords, valid, self.npoint)
        center_coords = torch.gather(coords, 1, center_idx.unsqueeze(-1).expand(-1, -1, coords.shape[-1]))
        group_idx = _knn_indices(center_coords, coords, valid, self.k)
        grouped_coords = _gather_points(coords, group_idx)
        grouped_features = _gather_points(features, group_idx)
        grouped_valid = _gather_mask(valid, group_idx) & center_valid.unsqueeze(-1)
        local = torch.cat([grouped_coords - center_coords.unsqueeze(2), grouped_features], dim=-1)
        local = self.local_mlp(local, grouped_valid)
        new_features = _masked_max(local, grouped_valid, dim=2)
        new_features = new_features.masked_fill(~center_valid.unsqueeze(-1), 0.0)
        return center_coords, new_features, center_valid


class GlobalSetAbstraction2D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        mid_dim = max(output_dim // 2, 64)
        self.mlp = MaskedSharedMLP([input_dim + 2, mid_dim, output_dim])

    def forward(self, coords, features, valid):
        centroid = _masked_mean(coords, valid, dim=1)
        local = torch.cat([coords - centroid.unsqueeze(1), features], dim=-1)
        h = self.mlp(local, valid)
        return _masked_max(h, valid, dim=1)


class EntityPointNet2(nn.Module):
    """2D PointNet++ style set abstraction over arc midpoints."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        embedding_dim=128,
        output_dim=5,
        pool="max",
        k=16,
        dropout=0.0,
    ):
        super().__init__()
        del pool
        self.sa1 = SetAbstraction2D(input_dim=input_dim, output_dim=hidden_dim, npoint=32, k=k)
        global_dim = max(hidden_dim * 4, embedding_dim)
        self.sa_global = GlobalSetAbstraction2D(input_dim=hidden_dim, output_dim=global_dim)
        self.rho = nn.Sequential(
            nn.Linear(global_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = MLP(embedding_dim, embedding_dim, output_dim) if output_dim is not None else None

    def encode(self, edge_sets):
        x, lengths = _to_padded(edge_sets)
        valid = _lengths_to_mask(lengths, x.shape[1])
        coords = x[..., :2]
        coords, features, valid = self.sa1(coords, x, valid)
        return self.rho(self.sa_global(coords, features, valid))

    def forward(self, edge_sets):
        z = self.encode(edge_sets)
        if self.head is None:
            return z
        return self.head(z)
