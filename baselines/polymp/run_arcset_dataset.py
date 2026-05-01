import argparse
import copy
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import yaml
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

POLYMP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POLYMP_ROOT.parents[1]
if str(POLYMP_ROOT) not in sys.path:
    sys.path.insert(0, str(POLYMP_ROOT))

from data.dataset import PolygonDataset
from model.nn import build_model
from train.trainer import Trainer

DATASETS = {
    "single_buildings": {
        "path": REPO_ROOT / "data" / "single_buildings" / "ShapeClassification.gpkg",
        "class_map": {"E": 0, "F": 1, "H": 2, "I": 3, "L": 4, "O": 5, "T": 6, "U": 7, "Y": 8, "Z": 9},
    },
    "single_mnist": {
        "path": REPO_ROOT / "data" / "single_mnist" / "mnist_scaled_normalized.gpkg",
        "class_map": {str(i): i for i in range(10)},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PolyMP on ArcSet datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model", choices=["polymp", "dsc_polymp"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="mps", help="mps | cpu | cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(requested):
    requested = requested.lower()
    if requested == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def is_mps_runtime_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = [
        "mps",
        "metal",
        "not implemented for",
        "unsupported device type",
        "sparse",
        "torch_scatter",
        "scatter",
    ]
    return any(needle in text for needle in needles)


def _iter_polygon_rings(geom):
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            yield poly
        return
    raise TypeError(f"Unsupported geometry type: {geom.geom_type}")


def polygon_to_polymp_arrays(geom):
    if geom.is_empty:
        raise ValueError("Empty geometry is not supported.")
    if not geom.is_valid:
        geom = make_valid(geom)

    contours = []
    rings = []
    for poly in _iter_polygon_rings(geom):
        exterior = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)
        if len(exterior) >= 3:
            rings.append(exterior)
            contours.append(len(exterior))
        for interior in poly.interiors:
            hole = np.asarray(interior.coords[:-1], dtype=np.float32)
            if len(hole) >= 3:
                rings.append(hole)
                contours.append(len(hole))

    if not rings:
        raise ValueError("Geometry does not contain a valid polygon ring.")

    pos = np.concatenate(rings, axis=0)

    src_parts = []
    dst_parts = []
    start = 0
    for count in contours:
        node_ids = np.arange(start, start + count, dtype=np.int64)
        src_parts.extend([node_ids, node_ids])
        dst_parts.extend([np.roll(node_ids, -1), np.roll(node_ids, 1)])
        start += count

    contour = np.stack([np.concatenate(src_parts), np.concatenate(dst_parts)], axis=0)
    return pos, contour, geom


def build_polymp_dataframe(gdf, class_map):
    rows = []
    for row in gdf.itertuples(index=False):
        label = str(row.label)
        if label not in class_map:
            raise KeyError(f"Label {label!r} is not in class map {sorted(class_map)}")
        pos, contour, geom = polygon_to_polymp_arrays(row.geometry)
        rows.append({
            "name": label,
            "pos": pos,
            "contour": contour,
            "trans": "o",
            "geom": geom.wkt,
        })
    return pd.DataFrame(rows)


def split_dataframe(df, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def load_base_cfg():
    with open(POLYMP_ROOT / "cfg" / "train_glyph.yaml", "r") as f:
        return yaml.safe_load(f)


def make_cfg(base_cfg, args, class_map, run_root, device):
    cfg = copy.deepcopy(base_cfg)
    cfg.update({
        "train": str(run_root / "train.pkl"),
        "val": str(run_root / "val.pkl"),
        "test": str(run_root / "test.pkl"),
        "cls": class_map,
        "epoch": args.epochs,
        "batch": args.batch,
        "worker": args.workers,
        "device": device,
        "out_channels": len(class_map),
        "nn": args.model,
        "model": args.model,
        "path": str(run_root / args.model),
    })
    return cfg


def make_loaders(train_df, val_df, test_df, cfg):
    dataset_kwargs = {"cls": cfg["cls"], "compute_spec": False}
    train_loader = DataLoader(
        PolygonDataset(train_df, **dataset_kwargs),
        batch_size=cfg["batch"],
        num_workers=cfg["worker"],
    )
    val_loader = DataLoader(
        PolygonDataset(val_df, **dataset_kwargs),
        batch_size=cfg["batch"],
        num_workers=cfg["worker"],
    )
    test_loader = DataLoader(
        PolygonDataset(test_df, **dataset_kwargs),
        batch_size=cfg["batch"],
        num_workers=cfg["worker"],
    )
    return train_loader, val_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    model.to(device)
    ys = []
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            ys.extend(data.y.detach().cpu().tolist())
            preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    acc = accuracy_score(ys, preds)
    macro_f1 = f1_score(ys, preds, average="macro")
    return acc, macro_f1


def fit_and_evaluate(cfg, train_loader, val_loader, test_loader):
    model = build_model(cfg)
    trainer = Trainer(cfg)
    trainer.fit(model, train_loader=train_loader, val_loader=val_loader)
    acc, macro_f1 = evaluate(model, test_loader, cfg["device"])
    return model, trainer, acc, macro_f1, cfg["device"]


def main():
    args = parse_args()
    dataset_cfg = DATASETS[args.dataset]
    run_root = POLYMP_ROOT / "_runs" / args.dataset
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_cfg['path']}")
    gdf = gpd.read_file(dataset_cfg["path"])
    df = build_polymp_dataframe(gdf, dataset_cfg["class_map"])
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)

    train_path = run_root / "train.pkl"
    val_path = run_root / "val.pkl"
    test_path = run_root / "test.pkl"
    train_df.to_pickle(train_path)
    val_df.to_pickle(val_path)
    test_df.to_pickle(test_path)

    requested_device = args.device
    device = resolve_device(requested_device)
    base_cfg = load_base_cfg()
    cfg = make_cfg(base_cfg, args, dataset_cfg["class_map"], run_root, device)
    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, cfg)

    print(
        f"Prepared {args.dataset}: train={len(train_df)} val={len(val_df)} "
        f"test={len(test_df)} model={args.model} device={device}"
    )

    try:
        _, _, acc, macro_f1, actual_device = fit_and_evaluate(cfg, train_loader, val_loader, test_loader)
    except Exception as exc:
        if device == "mps" and is_mps_runtime_error(exc):
            print(f"MPS execution failed, retrying on CPU: {type(exc).__name__}: {exc}")
            cfg = make_cfg(base_cfg, args, dataset_cfg["class_map"], run_root, "cpu")
            cfg["path"] = str(run_root / f"{args.model}_cpu_fallback")
            train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, cfg)
            _, _, acc, macro_f1, actual_device = fit_and_evaluate(cfg, train_loader, val_loader, test_loader)
        else:
            raise

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.epochs,
        "requested_device": requested_device,
        "actual_device": actual_device,
        "seed": args.seed,
        "class_map": dataset_cfg["class_map"],
        "counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "train_pickle": str(train_path),
        "val_pickle": str(val_path),
        "test_pickle": str(test_path),
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }
    summary_path = run_root / "summary.json"
    summary_model_path = run_root / f"summary_{args.model}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_model_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Summary saved to {summary_path}")
    print(f"Model summary saved to {summary_model_path}")


if __name__ == "__main__":
    main()
