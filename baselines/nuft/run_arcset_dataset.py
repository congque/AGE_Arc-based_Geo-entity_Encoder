"""
Adapter that runs the upstream NUFT (Mai et al., GeoInformatica 2023) shape-classification
training on our ArcSet GeoPackage datasets.

Pipeline
--------
1. Read our `.gpkg` with geopandas.
2. Convert each (Multi)Polygon to a fixed-vertex Shapely Polygon by:
     - exploding MultiPolygons (largest ring kept),
     - dropping the duplicated closing vertex,
     - resampling along the boundary to `--num_vert` evenly spaced points,
     - re-closing for shapely (last == first).
3. Build a GeoDataFrame with the columns the upstream `Trainer` expects:
     ID, geometry, geometry_norm, TYPEID, SPLIT_0   (1=train, 0=test, -1=valid)
   `geometry_norm` is rescaled into the (-1, 1, -1, 1) box.
4. Dump that gdf as a pickle in `--out_dir` and call upstream `Trainer` directly.

This keeps the NUFT codepath 100% intact (we only feed it the pickle it normally consumes
from DBSR-46K), so the model architecture, loss, optimiser, and arg semantics match the
authors' release.

Defaults match the smallest sensible NUFT-IFFT-MLP variant:
    pgon_enc      = nuft_ddsl    (pure NUFT spectral encoder + small FFN, no ResNet)
    nuft_freqXY   = 16 16        (256 complex coefficients)
    pgon_embed_dim= 64
    j             = 2            (polygon = 2-simplex mesh)
    batch_size    = 64
    cla_epoches   = 2            (smoke test)

Run
---
    python run_arcset_dataset.py \
        --gpkg ../../data/single_buildings/ShapeClassification.gpkg \
        --label_col label --num_vert 64 --num_epoch 2 --device mps

Set --device cpu / cuda / mps as appropriate.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import LinearRing
from sklearn.metrics import f1_score, accuracy_score
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POLYGONCODE = os.path.join(HERE, "polygoncode")
sys.path.insert(0, POLYGONCODE)


# --------------------------------------------------------------------------- #
# Geometry preprocessing
# --------------------------------------------------------------------------- #
def _largest_ring(geom):
    """Return the (Polygon) ring with the most exterior vertices from a
    Polygon or MultiPolygon."""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: len(p.exterior.coords))
    return geom


def resample_polygon(poly: Polygon, num_vert: int) -> Polygon:
    """Resample the exterior of `poly` to exactly `num_vert` unique vertices,
    evenly spaced by arc length. Returns a closed shapely Polygon with
    `num_vert + 1` coords (last == first)."""
    ring = LinearRing(poly.exterior.coords)
    L = ring.length
    # `num_vert` distinct samples in [0, L), then close with the first sample.
    ts = np.linspace(0.0, L, num_vert, endpoint=False)
    coords = np.array([ring.interpolate(t).coords[0] for t in ts])
    coords = np.vstack([coords, coords[0:1]])  # close
    return Polygon(coords)


def normalise_to_unit_box(poly: Polygon) -> Polygon:
    """Affine-rescale `poly` so its bounding box is (-1, -1, 1, 1), centred."""
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    s = max(w, h, 1e-9)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    coords = np.array(poly.exterior.coords)
    coords = (coords - np.array([cx, cy])) * (2.0 / s)
    return Polygon(coords)


def make_splits(n: int, seed: int = 0):
    """Return a length-n int array with values 1=train, -1=valid, 0=test
    using a 70/10/20 stratified-by-index split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    n_valid = int(0.1 * n)
    out = np.zeros(n, dtype=np.int64)
    out[idx[:n_train]] = 1
    out[idx[n_train : n_train + n_valid]] = -1
    out[idx[n_train + n_valid :]] = 0
    return out


# --------------------------------------------------------------------------- #
# Build upstream-format pickle
# --------------------------------------------------------------------------- #
def build_upstream_gdf(gpkg_path: str, label_col: str, num_vert: int, seed: int = 0):
    print(f"[adapter] loading {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    print(f"[adapter] {len(gdf)} rows; columns={list(gdf.columns)}")

    if label_col not in gdf.columns:
        # try common alternatives
        for cand in ("TYPEID", "label", "class", "type", "category"):
            if cand in gdf.columns:
                label_col = cand
                print(f"[adapter] using label column '{label_col}'")
                break
        else:
            raise SystemExit(f"no label column found in {gpkg_path}")

    # encode labels to a contiguous 0..K-1 int range
    raw_labels = np.asarray(gdf[label_col])
    classes = sorted(np.unique(raw_labels).tolist())
    cls2id = {c: i for i, c in enumerate(classes)}
    typeid = np.array([cls2id[v] for v in raw_labels], dtype=np.int64)
    print(f"[adapter] {len(classes)} classes -> {classes}")

    geoms_origin = []
    geoms_norm = []
    keep = []
    for i, g in enumerate(gdf.geometry):
        if g is None or g.is_empty:
            continue
        try:
            base = _largest_ring(g)
            if not isinstance(base, Polygon) or base.is_empty:
                continue
            r = resample_polygon(base, num_vert)
            n = normalise_to_unit_box(r)
            geoms_origin.append(r)
            geoms_norm.append(n)
            keep.append(i)
        except Exception as exc:  # noqa: BLE001
            print(f"[adapter] skip row {i}: {exc}")
            continue

    keep = np.asarray(keep)
    out = pd.DataFrame(
        {
            "ID": np.arange(len(keep), dtype=np.int64),
            "geometry": geoms_origin,
            "geometry_norm": geoms_norm,
            "TYPEID": typeid[keep],
            "SPLIT_0": make_splits(len(keep), seed=seed),
        }
    )
    out_gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)
    print(
        f"[adapter] kept {len(out_gdf)} rows; "
        f"split = train {(out_gdf.SPLIT_0 == 1).sum()}, "
        f"valid {(out_gdf.SPLIT_0 == -1).sum()}, "
        f"test {(out_gdf.SPLIT_0 == 0).sum()}"
    )
    return out_gdf


# --------------------------------------------------------------------------- #
# Train / eval using upstream Trainer
# --------------------------------------------------------------------------- #
def run_upstream_training(pickle_path: str, out_dir: str, args):
    # Defer imports until after sys.path fix.
    from polygonembed.trainer import Trainer, make_args_parser
    from polygonembed.data_util import load_dataframe

    upstream = make_args_parser()
    upstream_argv = [
        "--data_dir", out_dir + "/",
        "--model_dir", out_dir + "/model_dir/",
        "--log_dir", out_dir + "/model_dir/",
        "--pgon_filename", os.path.basename(pickle_path),
        "--geom_type_list", "norm",
        "--data_split_num", "0",
        "--task", "cla",
        "--model_type", "",
        "--pgon_enc", args.pgon_enc,
        "--pgon_embed_dim", str(args.pgon_embed_dim),
        "--nuft_freqXY", str(args.nuft_freqXY[0]), str(args.nuft_freqXY[1]),
        "--j", "2",
        "--padding_mode", "circular",
        "--do_polygon_random_start", "T",
        "--do_data_augment", "F",
        "--do_online_data_augment", "F",
        "--data_augment_type", "none",
        "--num_augment", "0",
        "--dropout", "0.1",
        "--spa_enc", "none",
        "--spa_embed_dim", "32",
        "--freq", "16",
        "--max_radius", "2",
        "--min_radius", "0.000001",
        "--spa_f_act", "relu",
        "--freq_init", "geometric",
        "--spa_enc_use_postmat", "F",
        "--k_delta", "1",
        "--num_hidden_layer", "1",
        "--hidden_dim", "128",
        "--use_layn", "T",
        "--skip_connection", "T",
        "--pgon_dec", "explicit_mlp",
        "--pgon_dec_grid_init", "uniform",
        "--pgon_dec_grid_enc_type", "none",
        "--grt_loss_func", "L2",
        "--do_weight_norm", "F",
        "--weight_decay", "0.0",
        "--pgon_norm_reg_weight", "0.02",
        "--task_loss_weight", "1.0",
        "--grt_epoches", "0",
        "--cla_epoches", str(args.num_epoch),
        "--log_every", "20",
        "--val_every", "9999999",
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--opt", "adam",
        "--act", "relu",
        "--balanced_train_loader", "F",
        "--device", args.device,
        "--tb", "F",
    ]
    parsed = upstream.parse_args(upstream_argv)
    pgon_gdf = load_dataframe(parsed.data_dir, parsed.pgon_filename)
    trainer = Trainer(parsed, pgon_gdf, console=True)
    t0 = time.time()
    trainer.run_train(save_eval=False)
    print(f"[adapter] train wall time: {time.time() - t0:.1f}s")
    # patched for ArcSet wrapper: run_train() already executes the upstream
    # final eval path, so avoid a redundant second reload/eval cycle here.
    return trainer, None


def evaluate_with_metrics(trainer, args):
    """Compute test accuracy + macro-F1 with sklearn (the upstream eval only
    prints accuracy)."""
    model = trainer.model.pgon_classifer
    if model is None:
        print("[adapter] no classifier; skipping metric evaluation.")
        return
    model.eval()
    loader = trainer.pgon_cla_dataloader["TEST"]
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            pgon_id, pgon_tensor, label = batch
            pgon_tensor = pgon_tensor.to(args.device)
            class_pred, _ = model.compute_class_pred(
                pgon_tensor,
                do_polygon_random_start=False,
                do_softmax=False,
            )
            pred = class_pred.argmax(dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(label.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print("=" * 60)
    print(f"[adapter] TEST accuracy = {acc:.4f}")
    print(f"[adapter] TEST macro-F1 = {f1:.4f}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpkg", required=True)
    p.add_argument("--label_col", default="label")
    p.add_argument("--num_vert", type=int, default=64)
    p.add_argument("--out_dir", default=None,
                   help="where to dump the upstream-format pickle and model_dir")
    p.add_argument("--num_epoch", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pgon_enc", default="nuft_ddsl",
                   choices=["nuft_ddsl", "nuftifft_ddsl", "nuft_specpool"])
    p.add_argument("--pgon_embed_dim", type=int, default=64)
    p.add_argument("--nuft_freqXY", nargs=2, type=int, default=[16, 16])
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | cuda:0 | mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.out_dir is None:
        ds_name = os.path.splitext(os.path.basename(args.gpkg))[0]
        args.out_dir = os.path.join(HERE, "_runs", ds_name)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "model_dir"), exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # MPS sanity: DDSL forces double precision, which MPS does not support;
    # fall back to CPU automatically when needed.
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("[adapter] MPS not available, falling back to CPU.")
        args.device = "cpu"
    if args.device == "mps":
        print("[adapter] WARNING: DDSL uses float64; MPS lacks float64 support."
              " Falling back to CPU for the NUFT block.")
        args.device = "cpu"

    pkl_name = "arcset_pgon.pkl"
    pkl_path = os.path.join(args.out_dir, pkl_name)
    if not os.path.exists(pkl_path):
        gdf = build_upstream_gdf(args.gpkg, args.label_col, args.num_vert, args.seed)
        with open(pkl_path, "wb") as f:
            pickle.dump(gdf, f)
        print(f"[adapter] wrote {pkl_path} ({os.path.getsize(pkl_path) / 1e6:.1f} MB)")
    else:
        print(f"[adapter] reusing cached {pkl_path}")

    trainer, _ = run_upstream_training(pkl_path, args.out_dir, args)
    evaluate_with_metrics(trainer, args)


if __name__ == "__main__":
    main()
