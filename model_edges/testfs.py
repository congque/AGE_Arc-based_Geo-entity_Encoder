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
import time

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .aux_stroke import ArcFeatureLayout, StrokeAuxiliaryHead, stroke_lambda
    from .entitydeepset import EntityDeepSet
    from .entitypointnet import EntityPointNet, EntityPointNet2
    from .entitysettransformer_sab import EntitySetTransformerSAB
    from .entitysettransformer_isab import EntitySetTransformerISAB
    from .load_points import load_gpkg_points
    from .load_entities import auto_num_freqs, count_arcs, geoms2sets
except ImportError:
    from aux_stroke import ArcFeatureLayout, StrokeAuxiliaryHead, stroke_lambda
    from entitydeepset import EntityDeepSet
    from entitypointnet import EntityPointNet, EntityPointNet2
    from entitysettransformer_sab import EntitySetTransformerSAB
    from entitysettransformer_isab import EntitySetTransformerISAB
    from load_points import load_gpkg_points
    from load_entities import auto_num_freqs, count_arcs, geoms2sets


DATASETS = {
    "single_buildings": ("data/single_buildings/ShapeClassification.gpkg", "label"),
    "single_mnist": ("data/single_mnist/mnist_scaled_normalized.gpkg", "label"),
    "single_omniglot": ("data/single_omniglot/omniglot.gpkg", "label"),
    "single_quickdraw": ("data/single_quickdraw/quickdraw.gpkg", "label"),
    # Per-entity isotropic-normalized variants (centroid + max(w,h)/2 -> [-1,1]).
    "single_buildings_iso": ("data/single_buildings/ShapeClassification_iso.gpkg", "label"),
    "single_mnist_iso": ("data/single_mnist/mnist_iso.gpkg", "label"),
    "single_omniglot_iso": ("data/single_omniglot/omniglot_iso.gpkg", "label"),
    "single_quickdraw_iso": ("data/single_quickdraw/quickdraw_iso.gpkg", "label"),
}

SET_MODELS = ["deepset", "pointnet", "pointnet2", "settransformer-sab", "settransformer-isab"]


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
    p.add_argument("--pointnet-pool", choices=["max", "mean", "max_mean"], default="max")
    p.add_argument("--pointnet-k", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--input-mode", choices=["arc", "points", "points_pe"], default="arc",
                   help="arc = ArcSet (segments). points = raw centred (x,y) per vertex. "
                        "points_pe = vertices + multi-frequency Fourier PE.")
    p.add_argument("--sab-pooling", choices=["mean", "pma"], default="mean",
                   help="set aggregation for SAB encoder; mean avoids PMA collapse on long arc sets")
    p.add_argument("--proto-distance", choices=["cosine", "sqeuclidean"], default="cosine",
                   help="distance metric for ProtoNet logits (default: cosine, bypasses LayerNorm norm-fix)")
    p.add_argument("--proto-init-temp", type=float, default=10.0,
                   help="initial temperature multiplier on logits (default 10)")
    p.add_argument("--proto-learnable-temp", action=argparse.BooleanOptionalAction, default=True,
                   help="make logit temperature a learnable parameter")
    p.add_argument("--decoder", choices=["proto", "lr"], default="proto",
                   help="test-time decoder; training stays ProtoNet for gradient flow")
    p.add_argument("--xy-num-freqs", default="auto")
    p.add_argument("--length-fourier", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--length-num-freqs", type=int, default=None)
    p.add_argument("--second-harmonic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-endpoints", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--aux-stroke", choices=["off", "on"], default="off")
    p.add_argument("--aux-mask-rate", type=float, default=0.3)
    p.add_argument("--aux-lambda-max", type=float, default=0.5)
    p.add_argument("--aux-warmup-epochs", type=int, default=20)
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


def load_episodic(path, label_column, xy_num_freqs, input_mode="arc", **fkwargs):
    gdf = gpd.read_file(path)
    geoms = gdf.geometry.tolist()
    if xy_num_freqs == "auto":
        avg = float(np.mean([count_arcs(g) for g in geoms]))
        xy_num_freqs = auto_num_freqs(avg)
        print(f"[load_episodic] avg arcs/entity = {avg:.2f} -> num_freqs = {xy_num_freqs}")

    if input_mode == "arc":
        edge_sets = geoms2sets(geoms, xy_num_freqs=xy_num_freqs, **fkwargs)
    else:
        # ignore arc-specific kwargs; use load_points helpers
        try:
            from .load_points import geom2points
        except ImportError:
            from load_points import geom2points
        mode = "raw" if input_mode == "points" else "pe"
        edge_sets = [geom2points(g, mode=mode, num_freqs=int(xy_num_freqs) if mode == "pe" else 0)
                     for g in geoms]
        print(f"[load_episodic] input_mode={input_mode} feature_dim={edge_sets[0].shape[1]}")

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


class ProtoHead(nn.Module):
    """ProtoNet decoder with optional learnable logit temperature.

    Two distance flavours:
      - cosine: logits = scale * cos(z_q, mean(z_sup_c))
      - sqeuclidean: logits = -scale * ||z_q - mean(z_sup_c)||^2

    LayerNorm in the SAB encoder fixes embedding norm to ~sqrt(D) and makes
    pairwise sq distances ~2D for unrelated pairs, so the differences between
    classes are dwarfed by the overall scale; softmax becomes flat and loss
    plateaus at chance even when argmax is mostly right. Cosine distance with
    a learnable scale is the standard fix and is the default here.
    """

    def __init__(self, distance: str = "cosine", init_temp: float = 10.0,
                 learnable: bool = True):
        super().__init__()
        self.distance = distance
        log_temp = torch.tensor(math.log(max(init_temp, 1e-6)), dtype=torch.float32)
        if learnable:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp)

    def forward(self, z_support: torch.Tensor, z_query: torch.Tensor,
                n_way: int, k_shot: int) -> torch.Tensor:
        prototypes = z_support.view(n_way, k_shot, -1).mean(dim=1)
        scale = self.log_temp.exp()
        if self.distance == "cosine":
            zq = F.normalize(z_query, dim=-1)
            zs = F.normalize(prototypes, dim=-1)
            return scale * (zq @ zs.t())
        diff = z_query.unsqueeze(1) - prototypes.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)
        return -scale * dist_sq


def proto_logits(z_support, z_query, n_way, k_shot, head: "ProtoHead | None" = None):
    if head is not None:
        return head(z_support, z_query, n_way, k_shot)
    # Backwards-compatible path used by older callers; equivalent to ProtoHead
    # with distance=sqeuclidean, scale=1.
    prototypes = z_support.view(n_way, k_shot, -1).mean(dim=1)
    diff = z_query.unsqueeze(1) - prototypes.unsqueeze(0)
    return -(diff ** 2).sum(dim=-1)


def lr_episode_metrics(z_support: torch.Tensor, z_query: torch.Tensor,
                       n_way: int, k_shot: int, query_labels: np.ndarray) -> dict[str, object]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("--decoder lr requires scikit-learn") from exc

    support_emb = F.normalize(z_support, dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)
    query_emb = F.normalize(z_query, dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)
    support_labels = np.repeat(np.arange(n_way, dtype=np.int64), k_shot)

    start = time.perf_counter()
    clf = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
    )
    clf.fit(support_emb, support_labels)
    query_probs = clf.predict_proba(query_emb)
    preds = clf.predict(query_emb)
    elapsed = time.perf_counter() - start

    classes = clf.classes_.astype(np.int64, copy=False)
    if query_probs.shape[1] != n_way or not np.array_equal(classes, np.arange(n_way)):
        full_probs = np.full((len(query_labels), n_way), 1e-12, dtype=np.float64)
        full_probs[:, classes] = query_probs
        query_probs = full_probs

    loss = float(-np.log(np.clip(query_probs[np.arange(len(query_labels)), query_labels], 1e-12, 1.0)).mean())
    acc = float(np.mean(preds == query_labels))
    return {
        "loss": loss,
        "acc": acc,
        "time_sec": float(elapsed),
    }


def to_tensor_batch(edge_sets, idx, device):
    return [torch.as_tensor(edge_sets[i], dtype=torch.float32, device=device) for i in idx]


def build_encoder(args, input_dim):
    common = dict(input_dim=input_dim, hidden_dim=args.hidden_dim,
                  embedding_dim=args.embedding_dim, output_dim=None)
    if args.set_model == "deepset":
        return EntityDeepSet(pool=args.pool, **common)
    if args.set_model == "pointnet":
        return EntityPointNet(pool=args.pointnet_pool, dropout=args.dropout, **common)
    if args.set_model == "pointnet2":
        return EntityPointNet2(pool=args.pointnet_pool, k=args.pointnet_k,
                               dropout=args.dropout, **common)
    if args.set_model == "settransformer-sab":
        return EntitySetTransformerSAB(num_heads=args.num_heads,
                                       num_encoder_blocks=args.num_encoder_blocks,
                                       num_decoder_blocks=args.num_decoder_blocks,
                                       set_pooling=args.sab_pooling,
                                       **common)
    if args.set_model == "settransformer-isab":
        return EntitySetTransformerISAB(num_heads=args.num_heads,
                                        num_encoder_blocks=args.num_encoder_blocks,
                                        num_decoder_blocks=args.num_decoder_blocks,
                                        num_inducing_points=args.num_inducing_points,
                                        set_pooling=args.sab_pooling,
                                        **common)
    raise ValueError(args.set_model)


def run_episodes(model, edge_sets, index_table, classes_pool, n_way, k_shot, n_query,
                 n_episodes, optimizer, device, rng, max_skips=None,
                 proto_head: "ProtoHead | None" = None,
                 aux_head: "StrokeAuxiliaryHead | None" = None,
                 aux_lambda: float = 0.0,
                 eval_decoder: str = "proto"):
    train = optimizer is not None
    model.train(train)
    if proto_head is not None:
        proto_head.train(train)
    if aux_head is not None:
        aux_head.train(train)
    losses, accs = [], []
    proto_losses = []
    aux_losses = []
    mdn_losses = []
    len_losses = []
    theta_losses = []
    mask_fracs = []
    masked_counts = []
    decoder_times = []
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
        episode_batch = sup_batch + qry_batch
        if train and aux_head is not None:
            z_all, aux_metrics = aux_head(model, episode_batch)
        else:
            z_all = model.encode(episode_batch)
            aux_metrics = None
        z_sup = z_all[:len(sup_batch)]
        z_qry = z_all[len(sup_batch):]
        if train or eval_decoder == "proto":
            logits = proto_logits(z_sup, z_qry, n_way, k_shot, head=proto_head)
            target = torch.from_numpy(qry_labels).to(device)
            proto_loss = F.cross_entropy(logits, target)
            loss = proto_loss
            if train and aux_metrics is not None:
                loss = loss + aux_lambda * aux_metrics["loss"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == target).float().mean().item()
            losses.append(float(loss.item()))
            proto_losses.append(float(proto_loss.item()))
        else:
            lr_metrics = lr_episode_metrics(z_sup, z_qry, n_way, k_shot, qry_labels)
            loss = lr_metrics["loss"]
            acc = lr_metrics["acc"]
            losses.append(loss)
            decoder_times.append(lr_metrics["time_sec"])
        accs.append(acc)
        if train and aux_metrics is not None:
            aux_losses.append(float(aux_metrics["loss"].item()))
            mdn_losses.append(float(aux_metrics["mdn_loss"].item()))
            len_losses.append(float(aux_metrics["length_loss"].item()))
            theta_losses.append(float(aux_metrics["theta_loss"].item()))
            mask_fracs.append(float(aux_metrics["mask_fraction"]))
            masked_counts.append(float(aux_metrics["masked_arcs"]))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
        "ci95": float(np.std(accs) / math.sqrt(max(len(accs), 1))) * 1.96 if accs else float("nan"),
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else float("nan"),
        "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
        "mdn_loss": float(np.mean(mdn_losses)) if mdn_losses else 0.0,
        "length_loss": float(np.mean(len_losses)) if len_losses else 0.0,
        "theta_loss": float(np.mean(theta_losses)) if theta_losses else 0.0,
        "mask_fraction": float(np.mean(mask_fracs)) if mask_fracs else 0.0,
        "masked_arcs": float(np.mean(masked_counts)) if masked_counts else 0.0,
        "decoder": "proto" if train or eval_decoder == "proto" else "lr",
        "decoder_time_mean_sec": float(np.mean(decoder_times)) if decoder_times else 0.0,
        "decoder_time_std_sec": float(np.std(decoder_times)) if decoder_times else 0.0,
        "decoder_times_sec": decoder_times,
        "episodes_completed": len(accs),
    }


def main():
    args = get_args()
    default_input, default_label = DATASETS[args.dataset]
    input_path = args.input or default_input
    label_column = args.label_column or default_label
    out_tag = args.set_model.replace("-", "_")
    decoder_suffix = "_lrhead" if args.decoder == "lr" else ""
    output_dir = args.output_dir or (
        f"model_edges/results/fs_{out_tag}_{args.dataset}_{args.n_way}w_{args.k_shot}s{decoder_suffix}")

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.input_mode == "arc":
        edge_sets, labels, split_col, used_xy_freqs = load_episodic(
            input_path,
            label_column,
            xy_num_freqs=parse_xy_num_freqs(args.xy_num_freqs),
            input_mode="arc",
            length_fourier=args.length_fourier,
            length_num_freqs=args.length_num_freqs,
            second_harmonic=args.second_harmonic,
            use_endpoints=args.use_endpoints,
        )
    else:
        edge_sets, labels, split_col, used_xy_freqs = load_episodic(
            input_path,
            label_column,
            xy_num_freqs=parse_xy_num_freqs(args.xy_num_freqs),
            input_mode=args.input_mode,
        )

    train_classes, val_classes, test_classes = class_splits(labels, split_col, args.seed)
    print(f"[classes] train={len(train_classes)} val={len(val_classes)} test={len(test_classes)}")

    all_classes = train_classes + val_classes + test_classes
    index_table = build_index(labels, all_classes)

    encoder = build_encoder(args, edge_sets[0].shape[1]).to(device)
    proto_head = ProtoHead(distance=args.proto_distance,
                           init_temp=args.proto_init_temp,
                           learnable=args.proto_learnable_temp).to(device)
    aux_head = None
    if args.aux_stroke == "on":
        if args.set_model in ("pointnet", "pointnet2"):
            raise ValueError(
                f"--aux-stroke on is not supported for {args.set_model}: "
                "the aux head reads encoder.stem / encoder.encoder / encoder.phi internals."
            )
        aux_head = StrokeAuxiliaryHead(
            feature_layout=ArcFeatureLayout(
                input_dim=edge_sets[0].shape[1],
                xy_num_freqs=used_xy_freqs,
                length_fourier=args.length_fourier,
                length_num_freqs=args.length_num_freqs,
                second_harmonic=args.second_harmonic,
                use_endpoints=args.use_endpoints,
            ),
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            mask_rate=args.aux_mask_rate,
        ).to(device)
    print(f"[model] {args.set_model} params={sum(p.numel() for p in encoder.parameters()):,}")
    print(f"[proto] distance={args.proto_distance} init_temp={args.proto_init_temp} "
          f"learnable={args.proto_learnable_temp}")
    print(f"[decoder] train=proto val=proto test={args.decoder}")
    if aux_head is not None:
        print(f"[aux] stroke-completion params={sum(p.numel() for p in aux_head.parameters()):,} "
              f"mask_rate={args.aux_mask_rate} lambda_max={args.aux_lambda_max} "
              f"warmup_epochs={args.aux_warmup_epochs}")
    else:
        print("[aux] disabled")
    params = list(encoder.parameters()) + list(proto_head.parameters())
    if aux_head is not None:
        params += list(aux_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_state = None
    best_val = -1.0
    best_epoch = 0

    best_proto_state = None
    best_aux_state = None
    history = []
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        aux_lambda = stroke_lambda(epoch, args.aux_lambda_max, args.aux_warmup_epochs) \
            if aux_head is not None else 0.0
        tr_metrics = run_episodes(encoder, edge_sets, index_table, train_classes,
                                  args.n_way, args.k_shot, args.n_query,
                                  args.train_episodes, optimizer, device, rng,
                                  proto_head=proto_head, aux_head=aux_head,
                                  aux_lambda=aux_lambda)
        with torch.no_grad():
            va_metrics = run_episodes(encoder, edge_sets, index_table, val_classes,
                                      args.n_way, args.k_shot, args.n_query,
                                      args.val_episodes, None, device, rng,
                                      proto_head=proto_head,
                                      eval_decoder="proto")
        epoch_time = time.perf_counter() - epoch_start
        cur_temp = float(proto_head.log_temp.exp().item())
        history_item = {
            "epoch": epoch,
            "train_loss": tr_metrics["loss"],
            "train_proto_loss": tr_metrics["proto_loss"],
            "train_aux_loss": tr_metrics["aux_loss"],
            "train_mdn_loss": tr_metrics["mdn_loss"],
            "train_length_loss": tr_metrics["length_loss"],
            "train_theta_loss": tr_metrics["theta_loss"],
            "train_acc": tr_metrics["acc"],
            "val_loss": va_metrics["loss"],
            "val_acc": va_metrics["acc"],
            "val_ci95": va_metrics["ci95"],
            "proto_temp": cur_temp,
            "aux_lambda": aux_lambda,
            "mask_fraction": tr_metrics["mask_fraction"],
            "masked_arcs": tr_metrics["masked_arcs"],
            "epoch_time_sec": epoch_time,
        }
        history.append(history_item)
        if aux_head is not None:
            print(f"epoch {epoch:02d} train_loss={tr_metrics['loss']:.4f} "
                  f"proto_loss={tr_metrics['proto_loss']:.4f} aux_loss={tr_metrics['aux_loss']:.4f} "
                  f"mdn={tr_metrics['mdn_loss']:.4f} len={tr_metrics['length_loss']:.4f} "
                  f"theta={tr_metrics['theta_loss']:.4f} train_acc={tr_metrics['acc']:.4f} "
                  f"val_acc={va_metrics['acc']:.4f} ±{va_metrics['ci95']:.4f} "
                  f"temp={cur_temp:.3f} aux_lambda={aux_lambda:.3f} "
                  f"mask_frac={tr_metrics['mask_fraction']:.3f} epoch_time={epoch_time:.2f}s")
        else:
            print(f"epoch {epoch:02d} train_loss={tr_metrics['loss']:.4f} train_acc={tr_metrics['acc']:.4f} "
                  f"val_acc={va_metrics['acc']:.4f} ±{va_metrics['ci95']:.4f} "
                  f"temp={cur_temp:.3f} epoch_time={epoch_time:.2f}s")
        if va_metrics["acc"] > best_val:
            best_val = va_metrics["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_proto_state = {k: v.detach().cpu().clone() for k, v in proto_head.state_dict().items()}
            if aux_head is not None:
                best_aux_state = {k: v.detach().cpu().clone() for k, v in aux_head.state_dict().items()}

    if best_state is not None:
        encoder.load_state_dict(best_state)
        if best_proto_state is not None:
            proto_head.load_state_dict(best_proto_state)
        if aux_head is not None and best_aux_state is not None:
            aux_head.load_state_dict(best_aux_state)
    with torch.no_grad():
        te_metrics = run_episodes(encoder, edge_sets, index_table, test_classes,
                                  args.n_way, args.k_shot, args.n_query,
                                  args.test_episodes, None, device, rng,
                                  proto_head=proto_head,
                                  eval_decoder=args.decoder)

    summary = {
        "dataset": args.dataset,
        "set_model": args.set_model,
        "decoder": args.decoder,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "n_query": args.n_query,
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": te_metrics["acc"],
        "test_ci95": te_metrics["ci95"],
        "test_loss": te_metrics["loss"],
        "device": str(device),
        "history": history,
        "decoder_metadata": {
            "train_decoder": "proto",
            "val_decoder": "proto",
            "test_decoder": args.decoder,
        },
        "config": {
            "xy_num_freqs": used_xy_freqs,
            "xy_num_freqs_mode": args.xy_num_freqs,
            "length_fourier": args.length_fourier,
            "second_harmonic": args.second_harmonic,
            "use_endpoints": args.use_endpoints,
            "sab_pooling": args.sab_pooling,
            "num_heads": args.num_heads,
            "num_encoder_blocks": args.num_encoder_blocks,
            "num_decoder_blocks": args.num_decoder_blocks,
            "num_inducing_points": args.num_inducing_points,
            "train_episodes": args.train_episodes,
            "val_episodes": args.val_episodes,
            "test_episodes": args.test_episodes,
            "proto_distance": args.proto_distance,
            "proto_init_temp": args.proto_init_temp,
            "proto_learnable_temp": args.proto_learnable_temp,
            "proto_final_temp": float(proto_head.log_temp.exp().item()),
            "decoder": args.decoder,
            "aux_stroke": args.aux_stroke,
            "aux_mask_rate": args.aux_mask_rate,
            "aux_lambda_max": args.aux_lambda_max,
            "aux_warmup_epochs": args.aux_warmup_epochs,
        },
    }
    if args.decoder == "lr":
        summary["decoder_metadata"].update({
            "embedding_normalization": "l2",
            "lr_solver": "lbfgs",
            "lr_multi_class": "multinomial",
            "lr_max_iter": 1000,
            "test_decoder_time_mean_sec": te_metrics["decoder_time_mean_sec"],
            "test_decoder_time_std_sec": te_metrics["decoder_time_std_sec"],
            "test_decoder_times_sec": te_metrics["decoder_times_sec"],
            "test_episodes_completed": te_metrics["episodes_completed"],
        })
    if aux_head is not None:
        summary["config"].update(aux_head.aux_config())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if best_state is not None:
        torch.save(best_state, out / "best.pt")
    if best_proto_state is not None:
        torch.save(best_proto_state, out / "best_proto.pt")
    if aux_head is not None and best_aux_state is not None:
        torch.save(best_aux_state, out / "best_aux.pt")

    print()
    print(f"best_epoch: {best_epoch}  val_acc: {best_val:.4f}")
    print(f"test: acc={te_metrics['acc']:.4f} ±{te_metrics['ci95']:.4f} "
          f"loss={te_metrics['loss']:.4f}")
    if args.decoder == "lr":
        print(f"test decoder lr: {te_metrics['decoder_time_mean_sec'] * 1e3:.2f} ms/episode "
              f"± {te_metrics['decoder_time_std_sec'] * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
