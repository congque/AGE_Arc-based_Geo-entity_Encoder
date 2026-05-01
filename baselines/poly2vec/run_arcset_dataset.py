"""Adapter: train Poly2Vec on the ArcSet shape-classification benchmarks.

Loads one of our 4 .gpkg datasets (single_buildings / single_mnist / single_omniglot /
single_quickdraw), normalises the geometry to [-1, 1] inside the unit square, pads
to a fixed length, and trains the upstream Poly2Vec geometry encoder followed by
a small classification head.

Usage:
    python run_arcset_dataset.py --dataset single_buildings --epochs 2

Notes
-----
* The polygon path triangulates each polygon via the ``triangle`` C library, which
  needs a CPU tensor (and is not vectorised).  We therefore force the entire
  Fourier encoder to run on CPU (Poly2Vec's official code uses CUDA for the linear
  parts but the triangulation hop alone forces a sync, so MPS gives no measurable
  win).  The classifier head and optimiser stay on the requested ``--device``.
* Multilinestrings (omniglot, quickdraw) are flattened to a single padded
  polyline by separating components with the previous endpoint repeated; the FT
  is summed over each component (linearity of FT) inside Poly2Vec's
  ``polyline_encoder``.  Architecture supports them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from torch.utils.data import DataLoader, Dataset

# Make poly2vec models importable when running this file directly
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from models.poly2vec import Poly2Vec  # noqa: E402

DATASETS = {
    "single_buildings": {
        "path": "data/single_buildings/ShapeClassification.gpkg",
        "kind": "polygons",
        "num_classes": 10,
    },
    "single_mnist": {
        "path": "data/single_mnist/mnist_scaled_normalized.gpkg",
        "kind": "polygons",
        "num_classes": 10,
    },
    "single_omniglot": {
        "path": "data/single_omniglot/omniglot.gpkg",
        "kind": "polylines",
        "num_classes": 1623,
    },
    "single_quickdraw": {
        "path": "data/single_quickdraw/quickdraw.gpkg",
        "kind": "polylines",
        "num_classes": 100,
    },
}


# ---------------------------------------------------------------------------
# Geometry preprocessing
# ---------------------------------------------------------------------------

def _polygon_coords(geom):
    """Return the exterior coords of the largest polygon ring.

    Poly2Vec's polygon FT does CDT triangulation of one simple ring, so we
    fall back on the largest exterior in case of MultiPolygon.
    """
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    coords = np.asarray(geom.exterior.coords, dtype=np.float64)[:-1]  # drop closing vertex
    return coords


def _polyline_coords(geom):
    """Return the (concatenated) coordinates of a (multi)linestring."""
    if isinstance(geom, MultiLineString):
        # Concatenate components.  Poly2Vec's polyline_encoder iterates over
        # consecutive 2-point segments; if we just concatenate, we'd inject a
        # spurious connector segment between strokes.  Instead, we duplicate
        # the endpoint of one stroke as the start of the next, then mark these
        # connector segments by zeroing their length post-hoc.  Simpler: just
        # return list of components; the caller handles padding component-wise
        # by computing FT per-stroke and summing.
        return [np.asarray(g.coords, dtype=np.float64) for g in geom.geoms]
    return [np.asarray(geom.coords, dtype=np.float64)]


def _bbox_norm(coords_iter):
    """Normalise a list of (N,2) arrays into [-1, 1] using shared bbox."""
    flat = np.concatenate(coords_iter, axis=0) if isinstance(coords_iter, list) else coords_iter
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    if isinstance(coords_iter, list):
        return [2 * (c - mn) / span - 1 for c in coords_iter]
    return 2 * (coords_iter - mn) / span - 1


class PolygonDataset(Dataset):
    """Pad polygon exteriors to a max length M; return (coords, length, label)."""

    def __init__(self, geoms, labels, max_len):
        self.coords = np.zeros((len(geoms), max_len, 2), dtype=np.float32)
        self.lengths = np.zeros(len(geoms), dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        for i, g in enumerate(geoms):
            c = _polygon_coords(g)
            c = _bbox_norm(c)
            n = min(len(c), max_len)
            self.coords[i, :n] = c[:n]
            self.lengths[i] = n

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.coords[i], self.lengths[i], self.labels[i]


class PolylineDataset(Dataset):
    """Multilinestring stored as a list of components per item.

    Poly2Vec's ``polyline_encoder`` accepts a single padded polyline per item.
    Strokes (multilinestring) are summed: we pre-compute per-component FT in
    the model wrapper.  Here we keep the components as a python list so the
    collate function pads each component independently.
    """

    def __init__(self, geoms, labels, max_len):
        self.components = []
        self.labels = np.asarray(labels, dtype=np.int64)
        self.max_len = int(max_len)
        for g in geoms:
            comps = _polyline_coords(g)
            comps = _bbox_norm(comps)
            self.components.append(comps)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.components[i], self.labels[i]


def polygon_collate(batch):
    coords = torch.from_numpy(np.stack([b[0] for b in batch]))
    lengths = torch.from_numpy(np.stack([b[1] for b in batch]))
    labels = torch.from_numpy(np.stack([b[2] for b in batch]))
    return coords, lengths, labels


def polyline_collate(batch, max_len):
    """Flatten each multilinestring into a list-of-tensors so the encoder
    can be called once per component and the FT contributions summed.
    """
    items = []
    labels = []
    for comps, y in batch:
        comp_list = []
        len_list = []
        for c in comps:
            arr = np.zeros((max_len, 2), dtype=np.float32)
            n = min(len(c), max_len)
            arr[:n] = c[:n].astype(np.float32, copy=False)
            comp_list.append(arr)
            len_list.append(n)
        items.append((np.stack(comp_list), np.asarray(len_list, dtype=np.int64)))
        labels.append(y)
    labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return items, labels


# ---------------------------------------------------------------------------
# Model wrapper: Poly2Vec encoder + classification head
# ---------------------------------------------------------------------------

class ShapeClassifier(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        # Poly2Vec encoder lives on its own device (forced cpu for triangle()).
        self.encoder = Poly2Vec(args=args, device=args.encoder_device)
        # The encoder doesn't get .to(...) called recursively because the
        # inner GeometryFourierEncoder is a plain class.  Move sub-modules.
        self.encoder.nn.to(args.encoder_device)
        self.encoder.param_mag.to(args.encoder_device)
        self.encoder.param_phase.to(args.encoder_device)
        self.head = nn.Sequential(
            nn.Linear(args.d_out, args.d_out),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_out, num_classes),
        )
        self.encoder_device = args.encoder_device
        self.head_device = args.head_device
        self.head.to(self.head_device)

    def encode_polygon_batch(self, coords, lengths):
        coords = coords.to(self.encoder_device)
        lengths = lengths.to(self.encoder_device)
        emb = self.encoder(coords, lengths, dataset_type="polygons")
        return emb

    def encode_polyline_batch(self, items):
        # items: list of (np.ndarray (K,M,2), np.ndarray (K,))
        embs = []
        # Lazy access to the U/V meshgrid for empty-stroke fallback
        ft_zero_shape = self.encoder.ft_encoder.U.shape
        for comp_arr, len_arr in items:
            valid = len_arr >= 2
            if not valid.any():
                # Degenerate item: produce a zero FT.
                ft_sum = torch.zeros(
                    ft_zero_shape, dtype=torch.complex64, device=self.encoder_device
                )
            else:
                comps = torch.from_numpy(comp_arr[valid]).to(self.encoder_device)
                lens = torch.from_numpy(len_arr[valid]).to(self.encoder_device)
                # The official polyline_encoder consumes a batch and returns
                # one FT per polyline; we sum FT contributions over components.
                ft = self.encoder.ft_encoder.encode(comps, lens, dataset_type="polylines")
                if ft.dim() == 3:  # already (K, H, W)
                    ft_sum = ft.sum(dim=0)
                else:                # squeezed singleton (H, W)
                    ft_sum = ft
            ft_sum = ft_sum.unsqueeze(0)  # add batch dim
            B = ft_sum.shape[0]
            mag = torch.abs(ft_sum).reshape(B, -1)
            phase = torch.angle(ft_sum).reshape(B, -1)
            mag = self.encoder.param_mag(mag)
            phase = self.encoder.param_phase(phase)
            emb = self.encoder.nn(torch.cat([mag, phase], dim=1))
            embs.append(emb)
        return torch.cat(embs, dim=0)

    def forward(self, batch, kind):
        if kind == "polygons":
            coords, lengths, _ = batch
            emb = self.encode_polygon_batch(coords, lengths)
        elif kind == "polylines":
            items, _ = batch
            emb = self.encode_polyline_batch(items)
        else:
            raise ValueError(kind)
        emb = emb.to(self.head_device)
        return self.head(emb)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def split_indices(n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def make_loaders(name, args):
    spec = DATASETS[name]
    path = (PROJECT_ROOT / spec["path"]).resolve()
    print(f"[load] {path} ...", flush=True)
    gdf = gpd.read_file(path)
    if args.limit:
        gdf = gdf.iloc[: args.limit].reset_index(drop=True)
    geoms = gdf.geometry.tolist()
    classes, labels = np.unique(gdf[args.label_column].to_numpy(), return_inverse=True)
    print(f"[load] N={len(geoms)} classes={len(classes)} kind={spec['kind']}", flush=True)

    if spec["kind"] == "polygons":
        max_len = int(np.percentile([len(_polygon_coords(g)) for g in geoms], 99))
        max_len = max(max_len, 8)
        ds = PolygonDataset(geoms, labels, max_len=max_len)
        collate = polygon_collate
    else:
        # take the 99th percentile per component
        all_lens = []
        for g in geoms:
            for c in _polyline_coords(g):
                all_lens.append(len(c))
        max_len = int(np.percentile(all_lens, 99))
        max_len = max(max_len, 4)
        ds = PolylineDataset(geoms, labels, max_len=max_len)
        collate = lambda b: polyline_collate(b, max_len)

    train_idx, val_idx, test_idx = split_indices(len(ds), args.seed)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)

    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        spec["kind"],
        len(classes),
        max_len,
    )


def run_epoch(model, loader, optimizer, criterion, kind):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        labels = batch[-1].to(model.head_device)
        with torch.set_grad_enabled(train):
            logits = model(batch, kind)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total += labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    return total_loss / max(total, 1), correct / max(total, 1)


def build_args(cli):
    """Build the SimpleNamespace expected by Poly2Vec from the CLI args."""
    # n_freqs=10 -> grid is (2*10+1) x 10 = 21*10 = 210, matching d_input=210.
    # The fusion field defaults to "concat" (the only branch giving 2*d_input).
    cfg = SimpleNamespace(
        gfm_params={"w_min": 0.1, "w_max": 1.0, "n_freqs": cli.n_freqs},
        d_input=(2 * cli.n_freqs + 1) * cli.n_freqs,
        d_hid=cli.d_hid,
        d_out=cli.d_out,
        dropout=cli.dropout,
        fusion="concat",
        encoder_device=cli.encoder_device,
        head_device=cli.head_device,
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-freqs", type=int, default=10)
    parser.add_argument("--d-hid", type=int, default=100)
    parser.add_argument("--d-out", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap dataset size (smoke testing).")
    parser.add_argument("--encoder-device", default="cpu",
                        help="Device for the Fourier encoder. Polygon path runs "
                             "the `triangle` library on CPU, so cpu is the safe "
                             "choice.  Lines/polylines could use mps.")
    parser.add_argument("--head-device", default=None,
                        help="Device for the classifier head (default: mps if "
                             "available else cpu).")
    parser.add_argument("--output-dir", default=None)
    cli = parser.parse_args()

    if cli.head_device is None:
        if torch.backends.mps.is_available():
            cli.head_device = "mps"
        elif torch.cuda.is_available():
            cli.head_device = "cuda"
        else:
            cli.head_device = "cpu"

    print(f"[device] encoder={cli.encoder_device} head={cli.head_device}", flush=True)

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    train_loader, val_loader, test_loader, kind, num_classes, max_len = make_loaders(cli.dataset, cli)
    print(f"[loaders] kind={kind} max_len={max_len} num_classes={num_classes}", flush=True)

    encoder_args = build_args(cli)
    model = ShapeClassifier(encoder_args, num_classes=num_classes)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparams:,}", flush=True)

    optimizer = torch.optim.Adam(
        list(model.encoder.nn.parameters())
        + list(model.encoder.param_mag.parameters())
        + list(model.encoder.param_phase.parameters())
        + list(model.head.parameters()),
        lr=cli.lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    history = []
    for epoch in range(cli.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, kind)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion, kind)
        dt = time.time() - t0
        print(
            f"epoch {epoch:02d} | train loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s",
            flush=True,
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "elapsed_s": dt,
        })
        best_val = max(best_val, val_acc)

    test_loss, test_acc = run_epoch(model, test_loader, None, criterion, kind)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}", flush=True)

    summary = {
        "dataset": cli.dataset,
        "kind": kind,
        "num_classes": num_classes,
        "max_len": max_len,
        "epochs_run": cli.epochs,
        "best_val_accuracy": best_val,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "params": int(nparams),
        "encoder_args": vars(encoder_args),
        "history": history,
    }
    out_dir = Path(cli.output_dir or HERE / f"results/{cli.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[save] {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
