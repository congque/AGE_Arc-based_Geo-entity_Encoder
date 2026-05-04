"""Size-only baseline: per-entity scalar features (bbox, area, perimeter,
n_arcs, mean_arc_length, aspect_ratio) -> small MLP classifier.

Used to bound how much of a method's accuracy is attributable to absolute
geometric size signal alone. If size-only beats a learned model, the dataset
leaks scale information.

Usage:
    python baselines/size_only/run_size_only.py --dataset single_buildings
    python baselines/size_only/run_size_only.py --dataset single_buildings_iso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "single_buildings": ("data/single_buildings/ShapeClassification.gpkg", "label"),
    "single_buildings_iso": ("data/single_buildings/ShapeClassification_iso.gpkg", "label"),
    "single_mnist": ("data/single_mnist/mnist_scaled_normalized.gpkg", "label"),
    "single_mnist_iso": ("data/single_mnist/mnist_iso.gpkg", "label"),
    "single_omniglot": ("data/single_omniglot/omniglot.gpkg", "label"),
    "single_omniglot_iso": ("data/single_omniglot/omniglot_iso.gpkg", "label"),
}


def extract_size_features(geom):
    """Return the 7-d scalar feature vector used by the size-only baseline."""
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    area = float(geom.area)
    perim = float(geom.length)
    aspect = w / max(h, 1e-9)

    if hasattr(geom, "geoms"):
        parts = list(geom.geoms)
    else:
        parts = [geom]
    arc_lengths = []
    for part in parts:
        coords = None
        if hasattr(part, "exterior"):
            coords = np.asarray(part.exterior.coords)
            for hole in part.interiors:
                coords_hole = np.asarray(hole.coords)
                arc_lengths.extend(np.linalg.norm(np.diff(coords_hole, axis=0), axis=1).tolist())
        elif hasattr(part, "coords"):
            coords = np.asarray(part.coords)
        if coords is not None and len(coords) >= 2:
            arc_lengths.extend(np.linalg.norm(np.diff(coords, axis=0), axis=1).tolist())
    n_arcs = len(arc_lengths)
    mean_arc = float(np.mean(arc_lengths)) if arc_lengths else 0.0

    return np.array([w, h, area, perim, aspect, float(n_arcs), mean_arc], dtype=np.float32)


def load_dataset(name):
    path_rel, label_col = DATASETS[name]
    gdf = gpd.read_file(REPO_ROOT / path_rel)
    feats = np.stack([extract_size_features(g) for g in gdf.geometry])
    raw_labels = gdf[label_col].astype(str).to_numpy()
    classes, labels = np.unique(raw_labels, return_inverse=True)
    return feats, labels.astype(np.int64), classes


def split_data(n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


class SizeMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(self.bn(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feats, labels, classes = load_dataset(args.dataset)
    n_classes = len(classes)
    print(f"[size-only] dataset={args.dataset} N={len(feats)} classes={n_classes}")

    feats_t = torch.from_numpy(feats)
    labels_t = torch.from_numpy(labels)
    train_idx, val_idx, test_idx = split_data(len(feats), seed=args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SizeMLP(feats.shape[1], n_classes, hidden=args.hidden).to(device)
    print(f"[size-only] model params={sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    def make_loader(idx, shuffle):
        return DataLoader(
            TensorDataset(feats_t[idx], labels_t[idx]),
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    def run_epoch(loader, optimizer):
        train = optimizer is not None
        model.train(train)
        total_loss = 0.0
        total_n = 0
        correct = 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = crit(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        return total_loss / total_n, correct / total_n

    best_val = -1.0
    best_state = None
    best_epoch = 0
    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(train_loader, optim)
        va_loss, va_acc = run_epoch(val_loader, None)
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                  f"val_acc={va_acc:.4f}")
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(test_loader, None)
    print(f"[size-only] best_epoch={best_epoch} val_acc={best_val:.4f} test_acc={test_acc:.4f}")

    summary = {
        "dataset": args.dataset,
        "method": "size_only_mlp",
        "feature_dim": feats.shape[1],
        "feature_names": ["bbox_w", "bbox_h", "area", "perimeter", "aspect_ratio",
                          "n_arcs", "mean_arc_length"],
        "num_classes": n_classes,
        "n_total": len(feats),
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "config": vars(args),
    }
    out_dir = Path(args.output_dir or
                   REPO_ROOT / "baselines" / "size_only" / "results" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[size-only] saved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
