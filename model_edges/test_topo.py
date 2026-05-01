"""Pair-wise topological-relation classification on PolygonGNN pairs.

Architecture: shared ArcSet encoder (deepset / settransformer-{sab,isab}) with no
classification head, applied to both polygons in a pair. Pair features are
[z_a, z_b, |z_a - z_b|, z_a * z_b]; an MLP head maps these to relation logits.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .entitydeepset import EntityDeepSet, MLP
    from .entitysettransformer_sab import EntitySetTransformerSAB
    from .entitysettransformer_isab import EntitySetTransformerISAB
    from .load_entities import auto_num_freqs, count_arcs, geom2set
except ImportError:
    from entitydeepset import EntityDeepSet, MLP
    from entitysettransformer_sab import EntitySetTransformerSAB
    from entitysettransformer_isab import EntitySetTransformerISAB
    from load_entities import auto_num_freqs, count_arcs, geom2set


SET_MODELS = ["deepset", "settransformer-sab", "settransformer-isab"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/topo_polygongnn/pairs.pkl")
    p.add_argument("--set-model", choices=SET_MODELS, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-encoder-blocks", type=int, default=2)
    p.add_argument("--num-decoder-blocks", type=int, default=1)
    p.add_argument("--num-inducing-points", type=int, default=16)
    p.add_argument("--pool", choices=["sum", "sum_mean"], default="sum")
    p.add_argument("--xy-num-freqs", default="auto")
    p.add_argument("--length-fourier", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--length-num-freqs", type=int, default=None)
    p.add_argument("--second-harmonic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-endpoints", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def parse_xy(value):
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return "auto"
    return int(value)


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PairDataset(Dataset):
    def __init__(self, edge_sets_a, edge_sets_b, labels):
        self.a = edge_sets_a
        self.b = edge_sets_b
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.y[idx]


def pair_collate(batch):
    a, b, y = zip(*batch)
    return list(a), list(b), torch.tensor(y, dtype=torch.long)


def build_encoder(args, input_dim):
    common = dict(input_dim=input_dim, hidden_dim=args.hidden_dim,
                  embedding_dim=args.embedding_dim, output_dim=None)
    if args.set_model == "deepset":
        return EntityDeepSet(pool=args.pool, **common)
    if args.set_model == "settransformer-sab":
        return EntitySetTransformerSAB(num_heads=args.num_heads,
                                       num_encoder_blocks=args.num_encoder_blocks,
                                       num_decoder_blocks=args.num_decoder_blocks,
                                       **common)
    if args.set_model == "settransformer-isab":
        return EntitySetTransformerISAB(num_heads=args.num_heads,
                                        num_encoder_blocks=args.num_encoder_blocks,
                                        num_decoder_blocks=args.num_decoder_blocks,
                                        num_inducing_points=args.num_inducing_points,
                                        **common)
    raise ValueError(args.set_model)


class SiameseTopoModel(nn.Module):
    def __init__(self, encoder, embedding_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.head = MLP(4 * embedding_dim, hidden_dim, num_classes, num_layers=3)

    def forward(self, sets_a, sets_b):
        z_a = self.encoder.encode(sets_a)
        z_b = self.encoder.encode(sets_b)
        feat = torch.cat([z_a, z_b, (z_a - z_b).abs(), z_a * z_b], dim=-1)
        return self.head(feat)


def to_device_list(edge_sets, idx_or_list, device):
    return [torch.as_tensor(x, dtype=torch.float32, device=device) for x in idx_or_list]


def run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    n_correct = 0
    with torch.set_grad_enabled(train):
        for sets_a, sets_b, y in loader:
            sets_a = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in sets_a]
            sets_b = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in sets_b]
            y = y.to(device)
            logits = model(sets_a, sets_b)
            loss = F.cross_entropy(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * y.shape[0]
            total_n += y.shape[0]
            n_correct += (logits.argmax(dim=1) == y).sum().item()
    return total_loss / total_n, n_correct / total_n


def main():
    args = get_args()
    out_tag = args.set_model.replace("-", "_")
    output_dir = args.output_dir or f"model_edges/results/topo_{out_tag}"

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with open(args.input, "rb") as f:
        pairs = pickle.load(f)
    print(f"loaded {len(pairs)} pairs")

    geoms_a = [r["poly_a"] for r in pairs]
    geoms_b = [r["poly_b"] for r in pairs]
    labels_str = [r["label"] for r in pairs]

    arc_counts = np.array([count_arcs(g) for g in geoms_a + geoms_b])
    avg_arcs = float(arc_counts.mean())
    if parse_xy(args.xy_num_freqs) == "auto":
        xy_num_freqs = auto_num_freqs(avg_arcs)
        print(f"[load] avg arcs/polygon={avg_arcs:.2f} -> xy_num_freqs={xy_num_freqs}")
    else:
        xy_num_freqs = int(args.xy_num_freqs)

    fkwargs = dict(xy_num_freqs=xy_num_freqs,
                   length_fourier=args.length_fourier,
                   length_num_freqs=args.length_num_freqs,
                   second_harmonic=args.second_harmonic,
                   use_endpoints=args.use_endpoints)
    edge_sets_a = [geom2set(g, **fkwargs) for g in geoms_a]
    edge_sets_b = [geom2set(g, **fkwargs) for g in geoms_b]

    classes, y = np.unique(labels_str, return_inverse=True)
    print(f"num_classes={len(classes)}")

    n = len(y)
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(n * 0.7); n_va = int(n * 0.15)
    tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    def subset(ix):
        return PairDataset([edge_sets_a[i] for i in ix],
                           [edge_sets_b[i] for i in ix],
                           [int(y[i]) for i in ix])

    train_loader = DataLoader(subset(tr_idx), batch_size=args.batch_size, shuffle=True, collate_fn=pair_collate)
    val_loader = DataLoader(subset(va_idx), batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate)
    test_loader = DataLoader(subset(te_idx), batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate)

    input_dim = edge_sets_a[0].shape[1]
    encoder = build_encoder(args, input_dim)
    model = SiameseTopoModel(encoder, args.embedding_dim, len(classes)).to(device)
    print(f"[model] {args.set_model} params={sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_val = -1.0
    best_epoch = 0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, None, device)
        print(f"epoch {epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_acc={va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = run_epoch(model, test_loader, None, device)

    summary = {
        "set_model": args.set_model,
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": te_acc,
        "test_loss": te_loss,
        "num_classes": int(len(classes)),
        "num_pairs": int(n),
        "device": str(device),
        "config": {
            "xy_num_freqs": int(xy_num_freqs),
            "xy_num_freqs_mode": args.xy_num_freqs,
            "length_fourier": args.length_fourier,
            "second_harmonic": args.second_harmonic,
            "use_endpoints": args.use_endpoints,
            "num_heads": args.num_heads,
            "num_encoder_blocks": args.num_encoder_blocks,
            "num_decoder_blocks": args.num_decoder_blocks,
            "num_inducing_points": args.num_inducing_points,
        },
    }
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if best_state is not None:
        torch.save(best_state, out / "best.pt")
    print(f"\nbest_epoch={best_epoch} test_acc={te_acc:.4f}")


if __name__ == "__main__":
    main()
