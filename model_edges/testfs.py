"""Few-shot episodic training with a Prototypical Network decoder.

Encoder = ArcSet variant (deepset / settransformer-sab / settransformer-isab) with no
classification head -- forward returns the (B, embedding_dim) embedding via .encode().

Decoder = Prototypical Network: each support set forms a class prototype (mean
embedding), each query is classified by negative squared Euclidean distance.

Class-level splits: by default uses the dataset's `split` column when present
(background -> train, evaluation -> test, with a slice carved out for val). When no
split column exists, classes are split 70/15/15 by seed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .entitydeepset import EntityDeepSet
    from .entitysettransformer_sab import EntitySetTransformerSAB
    from .entitysettransformer_isab import EntitySetTransformerISAB
    from .load_entities import auto_num_freqs, count_arcs, geoms2sets
except ImportError:
    from entitydeepset import EntityDeepSet
    from entitysettransformer_sab import EntitySetTransformerSAB
    from entitysettransformer_isab import EntitySetTransformerISAB
    from load_entities import auto_num_freqs, count_arcs, geoms2sets


DATASETS = {
    "single_buildings": ("data/single_buildings/ShapeClassification.gpkg", "label"),
    "single_mnist": ("data/single_mnist/mnist_scaled_normalized.gpkg", "label"),
    "single_omniglot": ("data/single_omniglot/omniglot.gpkg", "label"),
    "single_quickdraw": ("data/single_quickdraw/quickdraw.gpkg", "label"),
}

SET_MODELS = ["deepset", "settransformer-sab", "settransformer-isab"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--set-model", choices=SET_MODELS, required=True)
    p.add_argument("--input", default=None)
    p.add_argument("--label-column", default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--train-episodes", type=int, default=200, help="episodes per epoch")
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--test-episodes", type=int, default=600)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
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


def parse_xy_num_freqs(value):
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


def load_episodic(path, label_column, xy_num_freqs, **fkwargs):
    gdf = gpd.read_file(path)
    geoms = gdf.geometry.tolist()
    if xy_num_freqs == "auto":
        avg = float(np.mean([count_arcs(g) for g in geoms]))
        xy_num_freqs = auto_num_freqs(avg)
        print(f"[load_episodic] avg arcs/entity = {avg:.2f} -> xy_num_freqs = {xy_num_freqs}")
    edge_sets = geoms2sets(geoms, xy_num_freqs=xy_num_freqs, **fkwargs)
    labels = gdf[label_column].astype(str).to_numpy()
    split = gdf["split"].astype(str).to_numpy() if "split" in gdf.columns else None
    return edge_sets, labels, split, int(xy_num_freqs)


def class_splits(labels, split_col, seed):
    """Return three lists of class names for train/val/test."""
    rng = np.random.default_rng(seed)
    if split_col is not None:
        bg = sorted({l for l, s in zip(labels, split_col) if s == "background"})
        ev = sorted({l for l, s in zip(labels, split_col) if s == "evaluation"})
        if bg and ev:
            bg = list(bg); rng.shuffle(bg)
            n_val = max(1, int(len(bg) * 0.1))
            return bg[n_val:], bg[:n_val], ev
    classes = sorted(set(labels.tolist()))
    rng.shuffle(classes)
    n = len(classes)
    n_tr = int(n * 0.7); n_va = int(n * 0.15)
    return classes[:n_tr], classes[n_tr:n_tr + n_va], classes[n_tr + n_va:]


def build_index(labels, classes):
    table = {c: [] for c in classes}
    for i, l in enumerate(labels):
        if l in table:
            table[l].append(i)
    return {c: np.asarray(v, dtype=np.int64) for c, v in table.items() if v}


def sample_episode(index_table, classes_pool, n_way, k_shot, n_query, rng):
    available = [c for c in classes_pool if len(index_table[c]) >= k_shot + n_query]
    if len(available) < n_way:
        return None
    chosen = rng.choice(len(available), size=n_way, replace=False)
    classes = [available[i] for i in chosen]
    support, query, query_labels = [], [], []
    for cls_idx, c in enumerate(classes):
        pool = index_table[c]
        pick = rng.choice(len(pool), size=k_shot + n_query, replace=False)
        sup_idx = pool[pick[:k_shot]]
        qry_idx = pool[pick[k_shot:]]
        support.append(sup_idx)
        query.append(qry_idx)
        query_labels.extend([cls_idx] * n_query)
    return (
        np.concatenate(support),
        np.concatenate(query),
        np.asarray(query_labels, dtype=np.int64),
    )


def proto_logits(z_support, z_query, n_way, k_shot):
    z_support = z_support.view(n_way, k_shot, -1).mean(dim=1)
    diff = z_query.unsqueeze(1) - z_support.unsqueeze(0)
    return -(diff ** 2).sum(dim=-1)


def to_tensor_batch(edge_sets, idx, device):
    return [torch.as_tensor(edge_sets[i], dtype=torch.float32, device=device) for i in idx]


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


def run_episodes(model, edge_sets, index_table, classes_pool, n_way, k_shot, n_query,
                 n_episodes, optimizer, device, rng, max_skips=None):
    train = optimizer is not None
    model.train(train)
    losses, accs = [], []
    skipped = 0
    if max_skips is None:
        max_skips = n_episodes * 4
    while len(losses) < n_episodes:
        ep = sample_episode(index_table, classes_pool, n_way, k_shot, n_query, rng)
        if ep is None:
            skipped += 1
            if skipped > max_skips:
                break
            continue
        sup_idx, qry_idx, qry_labels = ep
        sup_batch = to_tensor_batch(edge_sets, sup_idx, device)
        qry_batch = to_tensor_batch(edge_sets, qry_idx, device)
        z_sup = model.encode(sup_batch)
        z_qry = model.encode(qry_batch)
        logits = proto_logits(z_sup, z_qry, n_way, k_shot)
        target = torch.from_numpy(qry_labels).to(device)
        loss = F.cross_entropy(logits, target)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == target).float().mean().item()
        losses.append(float(loss.item()))
        accs.append(acc)
    return float(np.mean(losses)) if losses else float("nan"), \
           float(np.mean(accs)) if accs else float("nan"), \
           float(np.std(accs) / math.sqrt(max(len(accs), 1))) * 1.96 if accs else float("nan")


def main():
    args = get_args()
    default_input, default_label = DATASETS[args.dataset]
    input_path = args.input or default_input
    label_column = args.label_column or default_label
    out_tag = args.set_model.replace("-", "_")
    output_dir = args.output_dir or (
        f"model_edges/results/fs_{out_tag}_{args.dataset}_{args.n_way}w_{args.k_shot}s")

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    edge_sets, labels, split_col, used_xy_freqs = load_episodic(
        input_path,
        label_column,
        xy_num_freqs=parse_xy_num_freqs(args.xy_num_freqs),
        length_fourier=args.length_fourier,
        length_num_freqs=args.length_num_freqs,
        second_harmonic=args.second_harmonic,
        use_endpoints=args.use_endpoints,
    )

    train_classes, val_classes, test_classes = class_splits(labels, split_col, args.seed)
    print(f"[classes] train={len(train_classes)} val={len(val_classes)} test={len(test_classes)}")

    all_classes = train_classes + val_classes + test_classes
    index_table = build_index(labels, all_classes)

    encoder = build_encoder(args, edge_sets[0].shape[1]).to(device)
    print(f"[model] {args.set_model} params={sum(p.numel() for p in encoder.parameters()):,}")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    best_state = None
    best_val = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        tr_loss, tr_acc, _ = run_episodes(encoder, edge_sets, index_table, train_classes,
                                          args.n_way, args.k_shot, args.n_query,
                                          args.train_episodes, optimizer, device, rng)
        with torch.no_grad():
            va_loss, va_acc, va_ci = run_episodes(encoder, edge_sets, index_table, val_classes,
                                                  args.n_way, args.k_shot, args.n_query,
                                                  args.val_episodes, None, device, rng)
        print(f"epoch {epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_acc={va_acc:.4f} ±{va_ci:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    if best_state is not None:
        encoder.load_state_dict(best_state)
    with torch.no_grad():
        te_loss, te_acc, te_ci = run_episodes(encoder, edge_sets, index_table, test_classes,
                                              args.n_way, args.k_shot, args.n_query,
                                              args.test_episodes, None, device, rng)

    summary = {
        "dataset": args.dataset,
        "set_model": args.set_model,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "n_query": args.n_query,
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": te_acc,
        "test_ci95": te_ci,
        "test_loss": te_loss,
        "device": str(device),
        "config": {
            "xy_num_freqs": used_xy_freqs,
            "xy_num_freqs_mode": args.xy_num_freqs,
            "length_fourier": args.length_fourier,
            "second_harmonic": args.second_harmonic,
            "use_endpoints": args.use_endpoints,
            "num_heads": args.num_heads,
            "num_encoder_blocks": args.num_encoder_blocks,
            "num_decoder_blocks": args.num_decoder_blocks,
            "num_inducing_points": args.num_inducing_points,
            "train_episodes": args.train_episodes,
            "val_episodes": args.val_episodes,
            "test_episodes": args.test_episodes,
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if best_state is not None:
        torch.save(best_state, out / "best.pt")

    print()
    print(f"best_epoch: {best_epoch}  val_acc: {best_val:.4f}")
    print(f"test: acc={te_acc:.4f} ±{te_ci:.4f} loss={te_loss:.4f}")


if __name__ == "__main__":
    main()
