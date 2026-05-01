"""Geo2Vec adapter for ArcSet shape-classification benchmarks.

Pipeline:
  1.  Read one of the four ArcSet gpkg datasets (single_buildings, single_mnist,
      single_omniglot, single_quickdraw).
  2.  Build a per-entity Geo2Vec embedding by calling ``list2vec`` from the
      cloned upstream repo (shape branch only -- each entity is centered &
      scaled to [-1, 1] before the SDF MLP is trained).
  3.  Train a small MLP classifier on top of the produced embeddings for
      ``--cls-epochs`` epochs (default 80).

This script does **not** modify the upstream Geo2Vec repo; it just imports it
and supplies its own argparse Namespace so we can keep the upstream defaults
while making the batch size and worker settings practical on Apple Silicon.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make sure unsupported MPS ops fall back to CPU (e.g. some shapely-side ops
# are pure CPU, but a few torch ops we rely on -- e.g. F.l1_loss reduction --
# can hit unimplemented kernels on MPS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import torch

# We import after env is set up, and we expose the upstream package layout via
# sys.path so that ``from models.Geo2Vec import ...`` etc. resolves.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import geopandas as gpd  # noqa: E402

from runners.list2embedding import list2vec  # noqa: E402


# ----------------------------------------------------------------------------
#                              Dataset registry
# ----------------------------------------------------------------------------
# Path is resolved relative to the project root (parent of ``baselines/``).
PROJECT_ROOT = HERE.parent.parent

DATASETS = {
    "single_buildings": {
        "path": PROJECT_ROOT / "data" / "single_buildings" / "ShapeClassification.gpkg",
        "label_col": "label",
        "expected_geom": "Polygon",
        "num_classes_hint": 10,
    },
    "single_mnist": {
        "path": PROJECT_ROOT / "data" / "single_mnist" / "mnist_scaled_normalized.gpkg",
        "label_col": "label",
        "expected_geom": "Polygon",
        "num_classes_hint": 10,
    },
    "single_omniglot": {
        "path": PROJECT_ROOT / "data" / "single_omniglot" / "omniglot.gpkg",
        "label_col": "label",
        "expected_geom": "MultiLineString",
        "num_classes_hint": 1623,
    },
    "single_quickdraw": {
        "path": PROJECT_ROOT / "data" / "single_quickdraw" / "quickdraw.gpkg",
        "label_col": "label",
        "expected_geom": "MultiLineString",
        "num_classes_hint": 100,
    },
}

LABEL_CANDIDATES = ("label", "labels", "class", "category", "class_id", "y", "target")


def _detect_label_column(gdf: gpd.GeoDataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in gdf.columns:
            return c
    # Fall back to the first non-geometry column
    for c in gdf.columns:
        if c != gdf.geometry.name:
            return c
    raise RuntimeError("No label column found in GeoDataFrame")


def load_arcset_gpkg(path: Path, label_col: str | None):
    print(f"[load] reading {path}")
    gdf = gpd.read_file(path)
    if label_col is None:
        label_col = _detect_label_column(gdf)
    print(f"[load] using label column: {label_col!r}")
    geoms = list(gdf.geometry.values)
    raw_labels = gdf[label_col].astype(str).tolist()
    classes = sorted(set(raw_labels))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cls2idx[c] for c in raw_labels], dtype=np.int64)
    print(f"[load] N={len(geoms)} geoms, {len(classes)} classes, "
          f"first geom_type={geoms[0].geom_type}")
    return geoms, y, classes


# ----------------------------------------------------------------------------
#                          Geo2Vec arg construction
# ----------------------------------------------------------------------------
def make_geo2vec_args(device: str, *,
                      sdf_epochs: int = 80,
                      batch_size: int = 1024 * 4,
                      num_process: int = 10,
                      samples_perUnit_shape: int = 100,
                      point_sample_shape: int = 20,
                      uniformed_sample_perUnit_shape: int = 20) -> argparse.Namespace:
    """Build an args namespace matching upstream ``get_args`` defaults.

    We keep the upstream Geo2Vec defaults intact except for a smaller batch
    size and ``num_workers=0`` for macOS / MPS stability.
    """
    ns = argparse.Namespace()
    ns.file_path = ""
    ns.save_file_name = ""
    # Sampling -- Location
    ns.num_process = num_process
    ns.samples_perUnit_location = 4000
    ns.point_sample_location = 10
    ns.sample_band_width_location = 0.1
    ns.uniformed_sample_perUnit_location = 30
    # Sampling -- Shape
    ns.samples_perUnit_shape = samples_perUnit_shape
    ns.point_sample_shape = point_sample_shape
    ns.sample_band_width_shape = 0.1
    ns.uniformed_sample_perUnit_shape = uniformed_sample_perUnit_shape
    # Training -- shared
    ns.batch_size = batch_size
    ns.num_workers = 0
    # Training -- Location
    ns.epochs_location = sdf_epochs
    ns.num_layers_location = 8
    ns.z_size_location = 256
    ns.hidden_size_location = 256
    ns.num_freqs_location = 16
    ns.device = device
    ns.code_reg_weight_location = 0.0
    ns.weight_decay_location = 0.01
    ns.polar_fourier_location = False
    ns.log_sampling_location = False
    ns.training_ratio_location = 0.95
    # Training -- Shape
    ns.epochs_shape = sdf_epochs
    ns.num_layers_shape = 8
    ns.z_size_shape = 256
    ns.hidden_size_shape = 256
    ns.num_freqs_shape = 8
    ns.device_shape = device
    ns.code_reg_weight_shape = 1.0
    ns.weight_decay_shape = 0.01
    ns.polar_fourier_shape = False
    ns.log_sampling_shape = True
    ns.training_ratio_shape = 0.95
    # Testing
    ns.test_representation_location = True
    ns.visualSDF_location = False
    ns.test_representation_shape = True
    ns.visualSDF_shape = False
    return ns


# ----------------------------------------------------------------------------
#                         Classifier head (smoke test)
# ----------------------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(d_hidden, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15,
                  seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for cls in range(n_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall:
            scores.append(2.0 * precision * recall / (precision + recall))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def train_classifier(emb: np.ndarray, y: np.ndarray, n_classes: int,
                     device: str, *, epochs: int = 2, batch_size: int = 256,
                     lr: float = 1e-3, hidden: int = 256):
    tr_idx, va_idx, te_idx = split_indices(len(y))
    x = torch.from_numpy(emb.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    model = MLP(emb.shape[1], hidden, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = torch.nn.CrossEntropyLoss()

    def _eval(idxs):
        model.eval()
        with torch.no_grad():
            xb = x[idxs].to(device)
            yb = yt[idxs].to(device)
            preds = model(xb).argmax(-1)
            acc = (preds == yb).float().mean().item()
            preds_np = preds.cpu().numpy()
            y_np = yb.cpu().numpy()
            macro_f1 = macro_f1_score(y_np, preds_np, n_classes)
            return acc, macro_f1

    history = []
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(ep + 1)
        perm = rng.permutation(len(tr_idx))
        running = 0.0
        n_seen = 0
        t0 = time.perf_counter()
        for s in range(0, len(perm), batch_size):
            sub = tr_idx[perm[s:s + batch_size]]
            xb = x[sub].to(device)
            yb = yt[sub].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(sub)
            n_seen += len(sub)
        dt = time.perf_counter() - t0
        tr_acc, tr_f1 = _eval(tr_idx[:min(2048, len(tr_idx))])
        va_acc, va_f1 = _eval(va_idx)
        te_acc, te_f1 = _eval(te_idx)
        history.append({
            "epoch": ep + 1, "loss": running / max(n_seen, 1),
            "train_acc": tr_acc, "train_macro_f1": tr_f1,
            "val_acc": va_acc, "val_macro_f1": va_f1,
            "test_acc": te_acc, "test_macro_f1": te_f1,
            "epoch_seconds": dt,
        })
        print(f"[cls] ep {ep+1}/{epochs} loss={running/max(n_seen,1):.4f} "
              f"train={tr_acc:.4f} val={va_acc:.4f} test={te_acc:.4f} "
              f"test_f1={te_f1:.4f} "
              f"({dt:.2f}s)")
    return history


# ----------------------------------------------------------------------------
#                                   Main
# ----------------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--device", default="mps",
                   help="torch device: mps | cuda | cpu")
    p.add_argument("--max-entities", type=int, default=0,
                   help="if >0 sub-sample to this many geoms (smoke testing)")
    p.add_argument("--sdf-epochs", type=int, default=80,
                   help="SDF (Geo2Vec) training epochs")
    p.add_argument("--cls-epochs", type=int, default=80,
                   help="downstream classifier epochs")
    p.add_argument("--num-process", type=int, default=10,
                   help="multiprocessing workers for SDF sampling")
    p.add_argument("--batch-size", type=int, default=1024 * 4,
                   help="SDF training batch size")
    p.add_argument("--samples-perUnit-shape", type=int, default=100)
    p.add_argument("--point-sample-shape", type=int, default=20)
    p.add_argument("--uniformed-sample-perUnit-shape", type=int, default=20)
    p.add_argument("--cls-batch-size", type=int, default=256)
    p.add_argument("--save-emb", type=str, default="",
                   help="optional path to save the produced embedding npy")
    return p.parse_args()


def resolve_device(name: str) -> str:
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if name in ("mps", "cuda"):
        print(f"[device] {name} requested but unavailable; falling back to cpu")
    return "cpu"


def main():
    args = parse_cli()
    cfg = DATASETS[args.dataset]
    geoms, y, classes = load_arcset_gpkg(cfg["path"], cfg["label_col"])

    if args.max_entities and args.max_entities < len(geoms):
        geoms = geoms[:args.max_entities]
        y = y[:args.max_entities]
        print(f"[load] sub-sampled to {len(geoms)} geoms")

    device = resolve_device(args.device)
    print(f"[device] using {device}")

    def _build_g2v_args(run_device: str) -> argparse.Namespace:
        return make_geo2vec_args(
            device=run_device,
            sdf_epochs=args.sdf_epochs,
            batch_size=args.batch_size,
            num_process=args.num_process,
            samples_perUnit_shape=args.samples_perUnit_shape,
            point_sample_shape=args.point_sample_shape,
            uniformed_sample_perUnit_shape=args.uniformed_sample_perUnit_shape,
        )

    def _run_geo2vec(run_device: str):
        g2v_args = _build_g2v_args(run_device)
        emb = list2vec(
            Geolist=geoms,
            save_model_path=None,
            Geo_dim=g2v_args.z_size_shape,
            num_epoch=args.sdf_epochs,
            location_learning=False,
            shape_learning=True,
            save_file_name=None,
            args=g2v_args,
        )
        return g2v_args, emb

    print(f"[geo2vec] running list2vec on {len(geoms)} entities ...")
    t0 = time.perf_counter()
    try:
        g2v_args, emb = _run_geo2vec(device)
    except FloatingPointError as exc:
        if device != "mps":
            raise
        print(f"[device] {exc} Retrying Geo2Vec on cpu.")
        device = "cpu"
        g2v_args, emb = _run_geo2vec(device)
    sdf_seconds = time.perf_counter() - t0
    print(f"[geo2vec] embedding shape={emb.shape} in {sdf_seconds:.1f}s")

    # Geo2Vec_Model is built with n_poly = max_id + 2; the first `len(geoms)`
    # rows correspond to ids 0..N-1 (matching the order in `Geolist`). We slice
    # accordingly so the embedding aligns with `y`.
    emb = emb[: len(geoms)]
    if args.save_emb:
        np.save(args.save_emb, emb)
        print(f"[geo2vec] saved embedding to {args.save_emb}")

    n_classes = max(int(y.max()) + 1, len(classes))
    print(f"[cls] training MLP classifier ({n_classes} classes) for "
          f"{args.cls_epochs} epochs ...")
    history = train_classifier(
        emb, y, n_classes=n_classes, device=device,
        epochs=args.cls_epochs, batch_size=args.cls_batch_size,
    )

    print("\n=== Summary ===")
    print(f"dataset           : {args.dataset}")
    print(f"n_entities        : {len(geoms)}")
    print(f"n_classes         : {n_classes}")
    print(f"sdf_seconds_total : {sdf_seconds:.1f}")
    print(f"sdf_epochs        : {args.sdf_epochs}")
    print(f"sdf_sec_per_epoch : {sdf_seconds / max(args.sdf_epochs, 1):.1f}")
    print(f"emb_dim           : {emb.shape[1]}")
    for h in history:
        print(f"cls ep{h['epoch']:>2}: loss={h['loss']:.4f} "
              f"val_acc={h['val_acc']:.4f} test_acc={h['test_acc']:.4f} "
              f"test_f1={h['test_macro_f1']:.4f} "
              f"({h['epoch_seconds']:.2f}s)")
    if history:
        final = history[-1]
        print(f"final_test_acc    : {final['test_acc']:.4f}")
        print(f"final_test_macro_f1: {final['test_macro_f1']:.4f}")


if __name__ == "__main__":
    # macOS multiprocessing safety: list2vec calls
    # ``multiprocessing.set_start_method('spawn', force=True)`` itself, which
    # re-imports this module. The ``if __name__ == '__main__'`` guard below
    # prevents recursion.
    main()
