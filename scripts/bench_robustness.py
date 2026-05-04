"""Train-clean / test-perturbed evaluation for ArcSet checkpoints.

Loads a trained checkpoint, applies a test-time geometry perturbation to the
test split (rotation, reflection, uniform scale, vertex Gaussian noise, or
Douglas-Peucker simplification), and reports the test accuracy.

Usage:
    python scripts/bench_robustness.py \
        --checkpoint model_edges/results/std_pointnet_single_buildings_iso_s0/best.pt \
        --dataset single_buildings_iso \
        --set-model pointnet \
        --perturb rotate --magnitude 45 \
        --input-mode arc

Outputs JSON with {perturb, magnitude, clean_acc, perturbed_acc, delta}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model_edges"))

from entitydeepset import EntityDeepSet
from entitypointnet import EntityPointNet, EntityPointNet2
from entitysettransformer_sab import EntitySetTransformerSAB
from entitysettransformer_isab import EntitySetTransformerISAB
from load_entities import geom2set, count_arcs, auto_num_freqs
from load_points import geom2points

from test import DATASETS, collate_fn, run_epoch


def perturb_geom(geom, kind, magnitude, rng):
    """Apply a per-entity perturbation. Returns a new shapely geometry."""
    if kind == "rotate":
        angle = math.radians(magnitude)
        cos, sin = math.cos(angle), math.sin(angle)
        return affine_transform(geom, [cos, -sin, sin, cos, 0, 0])
    if kind == "reflect_x":
        return affine_transform(geom, [1, 0, 0, -1, 0, 0])
    if kind == "scale":
        s = magnitude
        return affine_transform(geom, [s, 0, 0, s, 0, 0])
    if kind == "noise":
        # Add Gaussian noise to each vertex with std = magnitude
        if isinstance(geom, (Polygon, MultiPolygon)):
            polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
            new_polys = []
            for poly in polys:
                ext = np.asarray(poly.exterior.coords) + rng.normal(0, magnitude, (len(poly.exterior.coords), 2))
                ext[-1] = ext[0]
                interiors = []
                for hole in poly.interiors:
                    h = np.asarray(hole.coords) + rng.normal(0, magnitude, (len(hole.coords), 2))
                    h[-1] = h[0]
                    interiors.append(h)
                new_polys.append(Polygon(ext, interiors))
            return MultiPolygon(new_polys) if len(new_polys) > 1 else new_polys[0]
        if isinstance(geom, (LineString, MultiLineString)):
            lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
            new_lines = []
            for line in lines:
                pts = np.asarray(line.coords) + rng.normal(0, magnitude, (len(line.coords), 2))
                new_lines.append(LineString(pts))
            return MultiLineString(new_lines) if len(new_lines) > 1 else new_lines[0]
        return geom
    if kind == "simplify":
        return geom.simplify(magnitude, preserve_topology=True)
    raise ValueError(f"Unknown perturb kind={kind!r}")


def featurize(geoms, input_mode, num_freqs=8):
    """Convert list of shapely geoms to per-entity feature tensors."""
    if input_mode == "arc":
        return [geom2set(g, xy_num_freqs=num_freqs, length_fourier=True,
                         length_num_freqs=None, second_harmonic=True,
                         use_endpoints=False) for g in geoms]
    if input_mode == "points":
        return [geom2points(g, mode="raw") for g in geoms]
    if input_mode == "points_pe":
        return [geom2points(g, mode="pe", num_freqs=num_freqs) for g in geoms]
    raise ValueError(input_mode)


def build_model(set_model, input_dim, output_dim, hidden_dim=128, embedding_dim=128):
    common = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                  embedding_dim=embedding_dim, output_dim=output_dim)
    if set_model == "deepset":
        return EntityDeepSet(pool="sum", **common)
    if set_model == "pointnet":
        return EntityPointNet(pool="max", dropout=0.0, **common)
    if set_model == "pointnet2":
        return EntityPointNet2(pool="max", k=16, dropout=0.0, **common)
    if set_model == "settransformer-sab":
        return EntitySetTransformerSAB(num_heads=4, num_encoder_blocks=2,
                                        num_decoder_blocks=1, set_pooling="mean",
                                        **common)
    if set_model == "settransformer-isab":
        return EntitySetTransformerISAB(num_heads=4, num_encoder_blocks=2,
                                         num_decoder_blocks=1, set_pooling="mean",
                                         num_inducing_points=16, **common)
    raise ValueError(set_model)


def split_data(edge_sets, labels, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return idx[n_train + n_val:]  # test indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--set-model", required=True,
                   choices=["deepset", "pointnet", "pointnet2",
                            "settransformer-sab", "settransformer-isab"])
    p.add_argument("--input-mode", default="arc",
                   choices=["arc", "points", "points_pe"])
    p.add_argument("--perturb", required=True,
                   choices=["clean", "rotate", "reflect_x", "scale", "noise", "simplify"])
    p.add_argument("--magnitude", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--xy-num-freqs", type=int, default=None,
                   help="Override Fourier freq; default = auto from avg arcs")
    args = p.parse_args()

    path, label_col = DATASETS[args.dataset]
    gdf = gpd.read_file(REPO_ROOT / path)
    geoms_clean = gdf.geometry.tolist()
    raw_labels = gdf[label_col].astype(str).to_numpy()
    classes, labels = np.unique(raw_labels, return_inverse=True)

    if args.xy_num_freqs is None:
        avg = float(np.mean([count_arcs(g) for g in geoms_clean]))
        num_freqs = auto_num_freqs(avg)
    else:
        num_freqs = args.xy_num_freqs

    # Apply perturbation to ALL geoms; we'll only use the test indices for eval
    rng = np.random.default_rng(args.seed)
    if args.perturb == "clean":
        geoms_eval = geoms_clean
    else:
        geoms_eval = [perturb_geom(g, args.perturb, args.magnitude, rng)
                      for g in geoms_clean]

    feats = featurize(geoms_eval, args.input_mode, num_freqs=num_freqs)
    test_idx = split_data(feats, labels, args.seed)
    test_data = [(feats[i], int(labels[i])) for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    in_dim = feats[0].shape[1]
    out_dim = len(classes)
    model = build_model(args.set_model, in_dim, out_dim)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    crit = nn.CrossEntropyLoss()
    metrics = run_epoch(model, test_loader, None, crit, device)
    print(f"[bench-robustness] dataset={args.dataset} model={args.set_model} "
          f"input_mode={args.input_mode} perturb={args.perturb} mag={args.magnitude} "
          f"test_acc={metrics['accuracy']:.4f}")

    out = {
        "dataset": args.dataset,
        "set_model": args.set_model,
        "input_mode": args.input_mode,
        "checkpoint": args.checkpoint,
        "perturb": args.perturb,
        "magnitude": args.magnitude,
        "test_accuracy": float(metrics["accuracy"]),
        "test_loss": float(metrics["loss"]),
        "test_macro_f1": float(metrics["macro_f1"]),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[bench-robustness] saved to {args.output}")


if __name__ == "__main__":
    main()
