from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

try:
    from .load_entities import edge_feature_dim
except ImportError:
    from load_entities import edge_feature_dim


@dataclass(frozen=True)
class ArcFeatureLayout:
    input_dim: int
    xy_num_freqs: int
    length_fourier: bool
    length_num_freqs: int | None
    second_harmonic: bool
    use_endpoints: bool
    theta_bins: int = 16

    def __post_init__(self):
        expected = edge_feature_dim(
            xy_num_freqs=self.xy_num_freqs,
            length_fourier=self.length_fourier,
            length_num_freqs=self.length_num_freqs,
            second_harmonic=self.second_harmonic,
            use_endpoints=self.use_endpoints,
        )
        if expected != self.input_dim:
            raise ValueError(f"feature dim mismatch: expected {expected}, got {self.input_dim}")

    @property
    def xy_dim(self) -> int:
        return 4 + 6 * self.xy_num_freqs

    @property
    def endpoint_dim(self) -> int:
        return self.xy_dim * 2 if self.use_endpoints else 0

    @property
    def length_dim(self) -> int:
        return 1 + 2 * (self.length_num_freqs or self.xy_num_freqs) if self.length_fourier else 1

    @property
    def length_index(self) -> int:
        return self.xy_dim + self.endpoint_dim

    @property
    def theta_index(self) -> int:
        return self.length_index + self.length_dim


def stroke_lambda(epoch: int, lambda_max: float, warmup_epochs: int) -> float:
    if lambda_max <= 0.0:
        return 0.0
    if warmup_epochs <= 1:
        return float(lambda_max)
    scale = min(max(epoch, 0) / float(max(warmup_epochs - 1, 1)), 1.0)
    return float(lambda_max * scale)


def _pad_with_mask(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    padded = pad_sequence(items, batch_first=True)
    lengths = torch.tensor([item.shape[0] for item in items], device=padded.device)
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] >= lengths[:, None]
    return padded, mask


def _sinusoidal_positions(indices: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("position encoding dim must be positive")
    if dim == 1:
        return indices.to(dtype=torch.float32).unsqueeze(-1)
    half = dim // 2
    device = indices.device
    scale = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(-scale * torch.arange(half, device=device, dtype=torch.float32))
    angles = indices.to(dtype=torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    pos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        pos = F.pad(pos, (0, 1))
    return pos


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).unsqueeze(-1).to(dtype=x.dtype)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / counts


def _encode_with_tokens(
    encoder: nn.Module, edge_sets: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(encoder, "stem") and hasattr(encoder, "encoder"):
        tokens = [encoder.stem(edge_set) for edge_set in edge_sets]
        x, mask = _pad_with_mask(tokens)
        lengths = (~mask).sum(dim=1)
        for block in encoder.encoder:
            x = block(x, mask)
        if getattr(encoder, "set_pooling", None) == "mean":
            z = encoder.rho(_masked_mean(x, mask))
        else:
            pooled = encoder.pma(x, mask)
            for block in encoder.decoder:
                pooled = block(pooled)
            z = encoder.rho(pooled[:, 0])
        return z, x, mask

    if hasattr(encoder, "phi") and hasattr(encoder, "pool"):
        tokens = [encoder.phi(edge_set) for edge_set in edge_sets]
        x, mask = _pad_with_mask(tokens)
        pooled = []
        for token_set in tokens:
            if encoder.pool == "sum_mean":
                pooled.append(torch.cat([token_set.sum(dim=0), token_set.mean(dim=0)], dim=0))
            else:
                pooled.append(token_set.sum(dim=0))
        z = encoder.rho(torch.stack(pooled, dim=0))
        return z, x, mask

    raise TypeError(f"unsupported encoder type: {type(encoder)!r}")


class StrokeAuxiliaryHead(nn.Module):
    def __init__(
        self,
        *,
        feature_layout: ArcFeatureLayout,
        hidden_dim: int,
        embedding_dim: int,
        mask_rate: float = 0.3,
        decoder_dim: int = 64,
        decoder_layers: int = 2,
        mdn_components: int = 10,
        theta_bins: int = 16,
    ):
        super().__init__()
        self.layout = feature_layout
        self.mask_rate = mask_rate
        self.decoder_dim = decoder_dim
        self.mdn_components = mdn_components
        self.theta_bins = theta_bins

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.query_pos = nn.Linear(decoder_dim, decoder_dim)
        self.memory_pos = nn.Linear(decoder_dim, decoder_dim)
        self.centroid_proj = nn.Sequential(
            nn.Linear(2, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.global_query_proj = nn.Linear(embedding_dim, decoder_dim)
        self.global_memory_proj = nn.Linear(embedding_dim, decoder_dim)
        self.memory_proj = nn.Linear(hidden_dim, decoder_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=4,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        self.mdn_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, mdn_components * 6),
        )
        self.length_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 2),
        )
        self.theta_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, theta_bins),
        )

    def aux_config(self) -> dict[str, int | float]:
        return {
            "aux_mask_rate": self.mask_rate,
            "aux_decoder_dim": self.decoder_dim,
            "aux_decoder_layers": len(self.decoder.layers),
            "aux_mdn_components": self.mdn_components,
            "aux_theta_bins": self.theta_bins,
        }

    def _sample_mask(self, num_arcs: int, device: torch.device) -> torch.Tensor:
        if num_arcs < 2 or self.mask_rate <= 0.0:
            return torch.zeros(num_arcs, dtype=torch.bool, device=device)
        mask = torch.rand(num_arcs, device=device) < self.mask_rate
        if not mask.any():
            mask[torch.randint(num_arcs, (1,), device=device)] = True
        if mask.all():
            mask[torch.randint(num_arcs, (1,), device=device)] = False
        return mask

    def _theta_bins(self, edge_targets: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(
            edge_targets[:, self.layout.theta_index],
            edge_targets[:, self.layout.theta_index + 1],
        )
        phase = torch.remainder(theta + math.pi, 2.0 * math.pi)
        bins = torch.floor(phase / (2.0 * math.pi) * self.theta_bins).to(dtype=torch.long)
        return bins.clamp_(min=0, max=self.theta_bins - 1)

    def _mask_batch(
        self, edge_sets: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]], float]:
        masked_sets: list[torch.Tensor] = []
        metadata: list[dict[str, torch.Tensor]] = []
        total_arcs = 0
        total_masked = 0

        for edge_set in edge_sets:
            device = edge_set.device
            num_arcs = edge_set.shape[0]
            arc_mask = self._sample_mask(num_arcs, device)
            visible_idx = (~arc_mask).nonzero(as_tuple=False).squeeze(1)
            masked_idx = arc_mask.nonzero(as_tuple=False).squeeze(1)
            visible_set = edge_set.index_select(0, visible_idx)
            masked_targets = edge_set.index_select(0, masked_idx) if masked_idx.numel() else edge_set[:0]
            centroid = visible_set[:, :2].mean(dim=0) if visible_set.shape[0] else edge_set[:, :2].mean(dim=0)

            masked_sets.append(visible_set)
            metadata.append(
                {
                    "visible_idx": visible_idx,
                    "masked_idx": masked_idx,
                    "centroid": centroid,
                    "midpoint": masked_targets[:, :2],
                    "length": masked_targets[:, self.layout.length_index],
                    "theta": self._theta_bins(masked_targets) if masked_targets.shape[0] else masked_idx,
                }
            )
            total_arcs += num_arcs
            total_masked += masked_idx.numel()

        mask_fraction = float(total_masked / max(total_arcs, 1))
        return masked_sets, metadata, mask_fraction

    def _mdn_nll(self, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        params = params.view(-1, self.mdn_components, 6)
        pi_logits = params[:, :, 0]
        mu = params[:, :, 1:3]
        log_sigma = params[:, :, 3:5].clamp(min=-7.0, max=5.0)
        rho = torch.tanh(params[:, :, 5]).clamp(min=-0.95, max=0.95)

        sigma = log_sigma.exp()
        target = target.unsqueeze(1)
        norm = (target - mu) / sigma
        z_term = norm.pow(2).sum(dim=-1) - 2.0 * rho * norm[:, :, 0] * norm[:, :, 1]
        rho_term = (1.0 - rho.pow(2)).clamp_min(1e-5)

        log_prob = (
            -math.log(2.0 * math.pi)
            - log_sigma.sum(dim=-1)
            - 0.5 * torch.log(rho_term)
            - 0.5 * z_term / rho_term
        )
        log_mix = F.log_softmax(pi_logits, dim=-1) + log_prob
        return -torch.logsumexp(log_mix, dim=-1).mean()

    def _length_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = pred[:, 0]
        log_var = pred[:, 1].clamp(min=-7.0, max=7.0)
        precision = torch.exp(-log_var)
        return (0.5 * (log_var + (target - mean).pow(2) * precision)).mean()

    def forward(
        self, encoder: nn.Module, edge_sets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        masked_sets, metadata, mask_fraction = self._mask_batch(edge_sets)
        embeddings, token_memory, token_mask = _encode_with_tokens(encoder, masked_sets)

        active = [idx for idx, item in enumerate(metadata) if item["masked_idx"].numel() > 0]
        zero = embeddings.sum() * 0.0
        stats: dict[str, torch.Tensor | float] = {
            "loss": zero,
            "mdn_loss": zero.detach(),
            "length_loss": zero.detach(),
            "theta_loss": zero.detach(),
            "masked_arcs": float(sum(item["masked_idx"].numel() for item in metadata)),
            "mask_fraction": mask_fraction,
        }
        if not active:
            return embeddings, stats

        memory_items: list[torch.Tensor] = []
        query_items: list[torch.Tensor] = []
        midpoint_targets: list[torch.Tensor] = []
        length_targets: list[torch.Tensor] = []
        theta_targets: list[torch.Tensor] = []

        for batch_idx in active:
            item = metadata[batch_idx]
            valid = ~token_mask[batch_idx]
            visible_idx = item["visible_idx"]
            masked_idx = item["masked_idx"]

            memory = self.memory_proj(token_memory[batch_idx, valid])
            memory = memory + self.memory_pos(_sinusoidal_positions(visible_idx, self.decoder_dim))
            global_memory = self.global_memory_proj(embeddings[batch_idx]).unsqueeze(0)
            memory_items.append(torch.cat([global_memory, memory], dim=0))

            query = self.mask_token.expand(1, masked_idx.numel(), -1).squeeze(0)
            query = query + self.query_pos(_sinusoidal_positions(masked_idx, self.decoder_dim))
            query = query + self.centroid_proj(item["centroid"]).unsqueeze(0)
            query = query + self.global_query_proj(embeddings[batch_idx]).unsqueeze(0)
            query_items.append(query)

            midpoint_targets.append(item["midpoint"])
            length_targets.append(item["length"])
            theta_targets.append(item["theta"])

        memory_pad, memory_pad_mask = _pad_with_mask(memory_items)
        query_pad, query_pad_mask = _pad_with_mask(query_items)
        decoded = self.decoder(
            query_pad,
            memory_pad,
            tgt_key_padding_mask=query_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        valid_queries = ~query_pad_mask
        decoded = decoded[valid_queries]

        midpoint_target = torch.cat(midpoint_targets, dim=0)
        length_target = torch.cat(length_targets, dim=0)
        theta_target = torch.cat(theta_targets, dim=0)

        mdn_loss = self._mdn_nll(self.mdn_head(decoded), midpoint_target)
        length_loss = self._length_loss(self.length_head(decoded), length_target)
        theta_loss = F.cross_entropy(self.theta_head(decoded), theta_target)
        aux_loss = mdn_loss + 0.5 * length_loss + theta_loss

        stats["loss"] = aux_loss
        stats["mdn_loss"] = mdn_loss.detach()
        stats["length_loss"] = length_loss.detach()
        stats["theta_loss"] = theta_loss.detach()
        return embeddings, stats
