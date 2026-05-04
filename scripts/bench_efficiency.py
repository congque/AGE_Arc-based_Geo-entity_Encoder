"""Wall-clock and memory measurements for ArcSet variants.

Reports for one (method × dataset × input_mode):
- preprocess time (gpkg → arc/point feature tensors)
- model parameters
- forward time per entity (avg over 1k entities, batch=128)
- peak GPU memory during forward
- average tokens per entity

Usage:
    python scripts/bench_efficiency.py \
        --dataset single_buildings_iso --set-model pointnet --input-mode arc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model_edges"))

from entitydeepset import EntityDeepSet
from entitypointnet import EntityPointNet, EntityPointNet2
from entitysettransformer_sab import EntitySetTransformerSAB
from entitysettransformer_isab import EntitySetTransformerISAB
from load_entities import geom2set, count_arcs, auto_num_freqs
from load_points import geom2points

from test import DATASETS, collate_fn


def featurize_with_timing(geoms, input_mode, num_freqs):
    t0 = time.perf_counter()
    if input_mode == "arc":
        feats = [geom2set(g, xy_num_freqs=num_freqs, length_fourier=True,
                          length_num_freqs=None, second_harmonic=True,
                          use_endpoints=False) for g in geoms]
    elif input_mode == "points":
        feats = [geom2points(g, mode="raw") for g in geoms]
    elif input_mode == "points_pe":
        feats = [geom2points(g, mode="pe", num_freqs=num_freqs) for g in geoms]
    else:
        raise ValueError(input_mode)
    return feats, time.perf_counter() - t0


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--set-model", required=True,
                   choices=["deepset", "pointnet", "pointnet2",
                            "settransformer-sab", "settransformer-isab"])
    p.add_argument("--input-mode", default="arc",
                   choices=["arc", "points", "points_pe"])
    p.add_argument("--num-eval", type=int, default=1024,
                   help="how many entities to time forward on")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    path, label_col = DATASETS[args.dataset]
    t_load = time.perf_counter()
    gdf = gpd.read_file(REPO_ROOT / path)
    geoms = gdf.geometry.tolist()
    raw_labels = gdf[label_col].astype(str).to_numpy()
    classes, labels = np.unique(raw_labels, return_inverse=True)
    load_seconds = time.perf_counter() - t_load

    avg_arcs = float(np.mean([count_arcs(g) for g in geoms]))
    num_freqs = auto_num_freqs(avg_arcs)

    feats, featurize_seconds = featurize_with_timing(geoms, args.input_mode, num_freqs)
    avg_tokens = float(np.mean([len(f) for f in feats]))

    in_dim = feats[0].shape[1]
    out_dim = len(classes)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    model = build_model(args.set_model, in_dim, out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    n = min(args.num_eval, len(feats))
    eval_data = [(feats[i], int(labels[i])) for i in range(n)]
    loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # warmup
    with torch.no_grad():
        for batch in loader:
            edge_sets = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in batch[0]]
            _ = model(edge_sets)
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    total = 0
    with torch.no_grad():
        for batch in loader:
            edge_sets = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in batch[0]]
            _ = model(edge_sets)
            total += len(edge_sets)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - t0

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                   if device.type == "cuda" else None)

    out = {
        "dataset": args.dataset,
        "set_model": args.set_model,
        "input_mode": args.input_mode,
        "n_entities_total": len(geoms),
        "load_gpkg_seconds": load_seconds,
        "featurize_seconds": featurize_seconds,
        "featurize_per_entity_us": featurize_seconds * 1e6 / len(geoms),
        "avg_tokens_per_entity": avg_tokens,
        "input_feature_dim": in_dim,
        "params": n_params,
        "forward_seconds": forward_seconds,
        "forward_n_entities": total,
        "forward_per_entity_us": forward_seconds * 1e6 / total,
        "peak_gpu_mb": peak_mem_mb,
        "device": str(device),
    }
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
