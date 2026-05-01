from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

try:
    from .entitydeepset import MLP
except ImportError:
    from entitydeepset import MLP


class MAB(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = MLP(dim, dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k, key_padding_mask=None):
        h, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.norm1(q + h)
        q = self.norm2(q + self.ff(q))
        return q


class ISAB(nn.Module):
    """Induced Set Attention Block.

    H = MAB(I, X)   inducing points attend over the full set
    Y = MAB(X, H)   set elements attend over inducing summaries
    Cost: O(N * m) instead of O(N^2).
    """

    def __init__(self, dim, num_heads, num_inducing_points=16):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing_points, dim))
        nn.init.xavier_uniform_(self.inducing)
        self.mab_iq = MAB(dim, num_heads)
        self.mab_xq = MAB(dim, num_heads)

    def forward(self, x, mask=None):
        i = self.inducing.expand(x.shape[0], -1, -1)
        h = self.mab_iq(i, x, key_padding_mask=mask)
        return self.mab_xq(x, h, key_padding_mask=None)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, num_heads)

    def forward(self, x, mask=None):
        seed = self.seed.expand(x.shape[0], -1, -1)
        return self.mab(seed, x, key_padding_mask=mask)


class _SeedSAB(nn.Module):
    """Self-attention applied to the small post-PMA seed sequence (no padding)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.mab = MAB(dim, num_heads)

    def forward(self, x):
        return self.mab(x, x, key_padding_mask=None)


class EntitySetTransformerISAB(nn.Module):
    """ArcSet-SetTransformer-ISAB: induced-point attention encoder, O(N*m) per block."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        embedding_dim=128,
        output_dim=5,
        num_heads=4,
        num_encoder_blocks=2,
        num_decoder_blocks=1,
        num_inducing_points=16,
    ):
        super().__init__()
        self.stem = MLP(input_dim, hidden_dim, hidden_dim)
        self.encoder = nn.ModuleList(
            [ISAB(hidden_dim, num_heads, num_inducing_points) for _ in range(num_encoder_blocks)]
        )
        self.pma = PMA(hidden_dim, num_heads, num_seeds=1)
        self.decoder = nn.ModuleList([_SeedSAB(hidden_dim, num_heads) for _ in range(num_decoder_blocks)])
        self.rho = MLP(hidden_dim, hidden_dim, embedding_dim)
        self.head = MLP(embedding_dim, embedding_dim, output_dim) if output_dim is not None else None

    def encode(self, edge_sets):
        x = [self.stem(edge_set) for edge_set in edge_sets]
        lengths = torch.tensor([len(edge_set) for edge_set in x], device=x[0].device)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.shape[1], device=x.device)[None, :] >= lengths[:, None]

        for block in self.encoder:
            x = block(x, mask)

        x = self.pma(x, mask)

        for block in self.decoder:
            x = block(x)

        x = x[:, 0]
        return self.rho(x)

    def forward(self, edge_sets):
        z = self.encode(edge_sets)
        if self.head is None:
            return z
        return self.head(z)
