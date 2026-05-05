from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .entitydeepset import EntityDeepSet
    from .entitypointnet import EntityPointNet, EntityPointNet2
    from .entitysettransformer_sab import EntitySetTransformerSAB
    from .entitysettransformer_isab import EntitySetTransformerISAB
    from .load_entities import load_gpkg
    from .load_points import load_gpkg_points
except ImportError:
    from entitydeepset import EntityDeepSet
    from entitypointnet import EntityPointNet, EntityPointNet2
    from entitysettransformer_sab import EntitySetTransformerSAB
    from entitysettransformer_isab import EntitySetTransformerISAB
    from load_entities import load_gpkg
    from load_points import load_gpkg_points


DATASETS = {
    "single_buildings": ("data/single_buildings/ShapeClassification.gpkg", "label"),
    "single_mnist": ("data/single_mnist/mnist_scaled_normalized.gpkg", "label"),
    "single_omniglot": ("data/single_omniglot/omniglot.gpkg", "label"),
    "single_quickdraw": ("data/single_quickdraw/quickdraw.gpkg", "label"),
    # Per-entity isotropic-normalized variants (centroid + max(w,h)/2 -> [-1,1]).
    # Use these for matched-protocol comparisons across baselines.
    "single_buildings_iso": ("data/single_buildings/ShapeClassification_iso.gpkg", "label"),
    "single_mnist_iso": ("data/single_mnist/mnist_iso.gpkg", "label"),
    "single_omniglot_iso": ("data/single_omniglot/omniglot_iso.gpkg", "label"),
    "single_quickdraw_iso": ("data/single_quickdraw/quickdraw_iso.gpkg", "label"),
}


SET_MODELS = ["deepset", "pointnet", "pointnet2", "settransformer-sab", "settransformer-isab"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--set-model", choices=SET_MODELS, required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--pool", choices=["sum", "sum_mean"], default="sum")
    parser.add_argument("--pointnet-pool", choices=["max", "mean", "max_mean"], default="max")
    parser.add_argument("--pointnet-k", type=int, default=16)
    parser.add_argument("--pointnet-input-transform",
                        choices=["2d", "full", "none"], default="2d",
                        help="PointNet T-Net mode: '2d' default (transform first 2 dims), "
                             "'full' (transform all input dims), 'none' (drop T-Net).")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-blocks", type=int, default=2)
    parser.add_argument("--num-decoder-blocks", type=int, default=1)
    parser.add_argument("--num-inducing-points", type=int, default=16,
                        help="ISAB only: number of learnable inducing points")
    parser.add_argument("--sab-pooling", choices=["mean", "pma"], default="mean",
                        help="SAB/ISAB set aggregation; mean avoids PMA collapse on long arc sets")
    parser.add_argument("--xy-num-freqs", default="auto",
                        help='int or "auto" (default: auto = clip(ceil(log2(avg_arcs))+3, 6, 9))')
    parser.add_argument("--length-fourier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length-num-freqs", type=int, default=None)
    parser.add_argument("--second-harmonic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-endpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-shot", type=int, default=None,
                        help="If set, subsample N training examples per class (low-shot regime)")
    parser.add_argument("--input-mode", choices=["arc", "points", "points_pe"], default="arc",
                        help="arc = ArcSet (segments). points = raw centred (x,y) per vertex. "
                             "points_pe = vertices + multi-frequency Fourier PE.")
    parser.add_argument("--reflect-aug", action="store_true",
                        help="Augment training data with reflect_x (doubles train set size)")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def collate_fn(batch):
    edge_sets, labels = zip(*batch)
    return list(edge_sets), torch.tensor(labels, dtype=torch.long)


def split_data(edge_sets, labels, seed, low_shot=None):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    if low_shot is not None:
        labels_arr = np.asarray(labels)
        chosen = []
        for c in np.unique(labels_arr[train_idx]):
            cls_pool = train_idx[labels_arr[train_idx] == c]
            take = min(low_shot, len(cls_pool))
            chosen.append(cls_pool[:take])
        train_idx = np.concatenate(chosen)
        rng.shuffle(train_idx)

    return (
        [(edge_sets[i], labels[i]) for i in train_idx],
        [(edge_sets[i], labels[i]) for i in val_idx],
        [(edge_sets[i], labels[i]) for i in test_idx],
    )


def macro_f1(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    f1s = []
    for c in sorted(set(labels.tolist()) | set(preds.tolist())):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def run_epoch(model, loader, optimizer, criterion, device):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    all_labels = []
    all_preds = []

    with torch.set_grad_enabled(train):
        for edge_sets, labels in loader:
            edge_sets = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in edge_sets]
            labels = labels.to(device)
            logits = model(edge_sets)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * labels.shape[0]
            total_n += labels.shape[0]
            all_labels.append(labels.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return {
        "loss": total_loss / total_n,
        "accuracy": float((labels == preds).mean()),
        "macro_f1": macro_f1(labels, preds),
    }


def build_model(args, input_dim, output_dim):
    if args.set_model == "deepset":
        return EntityDeepSet(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            output_dim=output_dim,
            pool=args.pool,
        )
    if args.set_model == "pointnet":
        return EntityPointNet(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            output_dim=output_dim,
            pool=args.pointnet_pool,
            dropout=args.dropout,
            input_transform=args.pointnet_input_transform,
        )
    if args.set_model == "pointnet2":
        return EntityPointNet2(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            output_dim=output_dim,
            pool=args.pointnet_pool,
            k=args.pointnet_k,
            dropout=args.dropout,
        )
    if args.set_model == "settransformer-sab":
        return EntitySetTransformerSAB(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            output_dim=output_dim,
            num_heads=args.num_heads,
            num_encoder_blocks=args.num_encoder_blocks,
            num_decoder_blocks=args.num_decoder_blocks,
            set_pooling=args.sab_pooling,
        )
    if args.set_model == "settransformer-isab":
        return EntitySetTransformerISAB(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            output_dim=output_dim,
            set_pooling=args.sab_pooling,
            num_heads=args.num_heads,
            num_encoder_blocks=args.num_encoder_blocks,
            num_decoder_blocks=args.num_decoder_blocks,
            num_inducing_points=args.num_inducing_points,
        )
    raise ValueError(args.set_model)


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_xy_num_freqs(value):
    if value is None:
        return "auto"
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return int(value)


def main():
    args = get_args()
    default_input, default_label = DATASETS[args.dataset]
    input_path = args.input or default_input
    label_column = args.label_column or default_label
    out_tag = args.set_model.replace("-", "_")
    output_dir = args.output_dir or f"model_edges/results/{out_tag}_{args.dataset}"

    device = pick_device(args.device)

    if args.input_mode == "arc":
        edge_sets, raw_labels, used_xy_freqs = load_gpkg(
            input_path,
            label_column,
            xy_num_freqs=parse_xy_num_freqs(args.xy_num_freqs),
            length_fourier=args.length_fourier,
            length_num_freqs=args.length_num_freqs,
            second_harmonic=args.second_harmonic,
            use_endpoints=args.use_endpoints,
        )
    else:
        mode = "raw" if args.input_mode == "points" else "pe"
        edge_sets, raw_labels, used_xy_freqs = load_gpkg_points(
            input_path,
            label_column,
            mode=mode,
            num_freqs=parse_xy_num_freqs(args.xy_num_freqs) if mode == "pe" else 0,
        )
        print(f"[load_points] mode={mode} feature_dim={edge_sets[0].shape[1]}")
    classes, labels = np.unique(raw_labels, return_inverse=True)
    train_data, val_data, test_data = split_data(edge_sets, labels, args.seed, low_shot=args.low_shot)

    if args.reflect_aug:
        # Build a reflected version of every training entity (deterministic
        # 2x augmentation) by re-running the loader on shapely-reflected
        # geometries. Test/val left untouched.
        from shapely.affinity import affine_transform
        import geopandas as gpd
        from load_entities import geom2set
        from load_points import geom2points
        gdf_aug = gpd.read_file(input_path)
        geoms_reflected = [affine_transform(g, [1, 0, 0, -1, 0, 0]) for g in gdf_aug.geometry]
        if args.input_mode == "arc":
            edge_sets_aug = [geom2set(g, xy_num_freqs=used_xy_freqs,
                                       length_fourier=args.length_fourier,
                                       length_num_freqs=args.length_num_freqs,
                                       second_harmonic=args.second_harmonic,
                                       use_endpoints=args.use_endpoints)
                              for g in geoms_reflected]
        else:
            mode = "raw" if args.input_mode == "points" else "pe"
            edge_sets_aug = [geom2points(g, mode=mode,
                                          num_freqs=used_xy_freqs if mode == "pe" else 0)
                              for g in geoms_reflected]
        # Get the train indices used by split_data (regenerate to find them)
        rng_aug = np.random.default_rng(args.seed)
        idx_aug = np.arange(len(labels))
        rng_aug.shuffle(idx_aug)
        n_train_aug = int(len(idx_aug) * 0.7)
        train_idx_aug = idx_aug[:n_train_aug]
        train_data = train_data + [(edge_sets_aug[i], int(labels[i])) for i in train_idx_aug]
        print(f"[reflect-aug] doubled train set: {len(train_data)} examples")

    print(f"[split] train={len(train_data)} val={len(val_data)} test={len(test_data)} "
          f"low_shot={args.low_shot}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(args, edge_sets[0].shape[1], len(classes)).to(device)
    print(f"[model] {args.set_model} params={sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = run_epoch(model, val_loader, None, criterion, device)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, None, criterion, device)

    summary = {
        "dataset": args.dataset,
        "set_model": args.set_model,
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test": test_metrics,
        "num_classes": int(len(classes)),
        "num_samples": int(len(labels)),
        "max_epochs": args.epochs,
        "input_dim": int(edge_sets[0].shape[1]),
        "device": str(device),
        "low_shot": args.low_shot,
        "config": {
            "pool": args.pool,
            "xy_num_freqs": used_xy_freqs,
            "xy_num_freqs_mode": args.xy_num_freqs,
            "length_fourier": args.length_fourier,
            "length_num_freqs": args.length_num_freqs,
            "second_harmonic": args.second_harmonic,
            "use_endpoints": args.use_endpoints,
            "num_heads": args.num_heads,
            "num_encoder_blocks": args.num_encoder_blocks,
            "num_decoder_blocks": args.num_decoder_blocks,
            "num_inducing_points": args.num_inducing_points,
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    torch.save(best_state, out_dir / "best.pt")

    print()
    print("best_epoch:", best_epoch)
    print("test:", test_metrics)


if __name__ == "__main__":
    main()
