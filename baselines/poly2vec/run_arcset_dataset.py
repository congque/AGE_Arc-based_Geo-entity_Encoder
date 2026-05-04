"""Adapter: train Poly2Vec on ArcSet using cached Fourier features.

The CPU-bound triangulation / Fourier preprocessing is performed once and saved
to disk. Subsequent runs train only the original Poly2Vec MLP stack plus the
classification head on the requested accelerator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import MultiLineString, MultiPolygon
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

# Make poly2vec models importable when running this file directly.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from models.poly2vec import Poly2Vec  # noqa: E402

DATASETS = {
    "single_buildings": {
        "path": PROJECT_ROOT / "data" / "single_buildings" / "ShapeClassification.gpkg",
        "kind": "polygons",
    },
    "single_mnist": {
        "path": PROJECT_ROOT / "data" / "single_mnist" / "mnist_scaled_normalized.gpkg",
        "kind": "polygons",
    },
    "single_omniglot": {
        "path": PROJECT_ROOT / "data" / "single_omniglot" / "omniglot.gpkg",
        "kind": "polylines",
    },
    "single_quickdraw": {
        "path": PROJECT_ROOT / "data" / "single_quickdraw" / "quickdraw.gpkg",
        "kind": "polylines",
    },
    # Per-entity isotropic-normalized variants (centroid + max(w,h)/2 -> [-1,1]).
    "single_buildings_iso": {
        "path": PROJECT_ROOT / "data" / "single_buildings" / "ShapeClassification_iso.gpkg",
        "kind": "polygons",
    },
    "single_mnist_iso": {
        "path": PROJECT_ROOT / "data" / "single_mnist" / "mnist_iso.gpkg",
        "kind": "polygons",
    },
    "single_omniglot_iso": {
        "path": PROJECT_ROOT / "data" / "single_omniglot" / "omniglot_iso.gpkg",
        "kind": "polylines",
    },
    "single_quickdraw_iso": {
        "path": PROJECT_ROOT / "data" / "single_quickdraw" / "quickdraw_iso.gpkg",
        "kind": "polylines",
    },
}

LABEL_CANDIDATES = ("label", "labels", "class", "category", "class_id", "y", "target")


def _detect_label_column(gdf: gpd.GeoDataFrame, preferred: str | None) -> str:
    if preferred and preferred in gdf.columns:
        return preferred
    for col in LABEL_CANDIDATES:
        if col in gdf.columns:
            return col
    for col in gdf.columns:
        if col != gdf.geometry.name:
            return col
    raise RuntimeError("No label column found in GeoDataFrame")


def _polygon_coords(geom) -> np.ndarray:
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda poly: poly.area)
    return np.asarray(geom.exterior.coords, dtype=np.float64)[:-1]


def _polyline_coords(geom) -> list[np.ndarray]:
    if isinstance(geom, MultiLineString):
        return [np.asarray(part.coords, dtype=np.float64) for part in geom.geoms]
    return [np.asarray(geom.coords, dtype=np.float64)]


def _bbox_norm(coords_iter):
    flat = np.concatenate(coords_iter, axis=0) if isinstance(coords_iter, list) else coords_iter
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    if isinstance(coords_iter, list):
        return [2.0 * (coords - mn) / span - 1.0 for coords in coords_iter]
    return 2.0 * (coords_iter - mn) / span - 1.0


def detect_geometry_kind(geoms, fallback: str) -> str:
    geom_types = {geom.geom_type for geom in geoms}
    if geom_types <= {"Polygon", "MultiPolygon"}:
        return "polygons"
    if geom_types <= {"LineString", "MultiLineString"}:
        return "polylines"
    return fallback


def resolve_device(name: str | None) -> str:
    requested = name or "cuda"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if requested in {"cuda", "mps"}:
        print(f"[device] {requested} requested but unavailable; falling back to cpu", flush=True)
    return "cpu"


def synchronize_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def split_indices(n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def build_args(cli, train_device: str) -> SimpleNamespace:
    return SimpleNamespace(
        gfm_params={"w_min": 0.1, "w_max": 1.0, "n_freqs": cli.n_freqs},
        d_input=(2 * cli.n_freqs + 1) * cli.n_freqs,
        d_hid=cli.d_hid,
        d_out=cli.d_out,
        dropout=cli.dropout,
        fusion="concat",
        encoder_device="cpu",
        head_device=train_device,
    )


def default_feature_cache_path(cli) -> Path:
    limit_suffix = f"_limit{cli.limit}" if cli.limit else ""
    kind = DATASETS[cli.dataset]["kind"]
    return HERE / "cache" / cli.dataset / (
        f"cache_{cli.dataset}_{kind}_nfreqs{cli.n_freqs}{limit_suffix}.pt"
    )


def load_dataset(cli):
    spec = DATASETS[cli.dataset]
    print(f"[load] reading {spec['path']}", flush=True)
    gdf = gpd.read_file(spec["path"])
    label_column = _detect_label_column(gdf, cli.label_column)
    if cli.limit:
        gdf = gdf.iloc[: cli.limit].reset_index(drop=True)
    geoms = gdf.geometry.tolist()
    labels_raw = gdf[label_column].astype(str).to_numpy()
    classes, labels = np.unique(labels_raw, return_inverse=True)
    kind = detect_geometry_kind(geoms, spec["kind"])
    print(
        f"[load] N={len(geoms)} classes={len(classes)} kind={kind} label_column={label_column}",
        flush=True,
    )
    return geoms, torch.from_numpy(labels.astype(np.int64)), [str(cls) for cls in classes.tolist()], kind, label_column


def encode_polygon_feature(ft_encoder, geom, expected_dim: int) -> torch.Tensor:
    coords = _bbox_norm(_polygon_coords(geom)).astype(np.float32, copy=False)
    coords_t = torch.from_numpy(coords).unsqueeze(0)
    lengths_t = torch.tensor([coords.shape[0]], dtype=torch.long)
    ft = ft_encoder.encode(coords_t, lengths_t, dataset_type="polygons").reshape(-1)
    feature = torch.cat([torch.abs(ft), torch.angle(ft)], dim=0).to(torch.float32)
    if feature.numel() != expected_dim:
        raise RuntimeError(f"Polygon feature dim mismatch: got {feature.numel()}, expected {expected_dim}")
    return feature.cpu()


def encode_polyline_feature(ft_encoder, geom, expected_dim: int) -> torch.Tensor:
    comps = _bbox_norm(_polyline_coords(geom))
    valid = [comp.astype(np.float32, copy=False) for comp in comps if len(comp) >= 2]
    if not valid:
        ft_sum = torch.zeros_like(ft_encoder.U, dtype=torch.complex64).reshape(-1)
        feature = torch.cat([torch.abs(ft_sum), torch.angle(ft_sum)], dim=0).to(torch.float32)
        return feature.cpu()

    max_len = max(len(comp) for comp in valid)
    comp_tensor = torch.zeros((len(valid), max_len, 2), dtype=torch.float32)
    lengths = torch.empty(len(valid), dtype=torch.long)
    for idx, comp in enumerate(valid):
        comp_tensor[idx, : len(comp)] = torch.from_numpy(comp)
        lengths[idx] = len(comp)

    ft = ft_encoder.encode(comp_tensor, lengths, dataset_type="polylines")
    ft_sum = ft.sum(dim=0) if ft.dim() > 2 else ft
    feature = torch.cat([torch.abs(ft_sum.reshape(-1)), torch.angle(ft_sum.reshape(-1))], dim=0).to(torch.float32)
    if feature.numel() != expected_dim:
        raise RuntimeError(f"Polyline feature dim mismatch: got {feature.numel()}, expected {expected_dim}")
    return feature.cpu()


def build_feature_cache(cli, encoder_args: SimpleNamespace, cache_path: Path):
    geoms, labels, classes, kind, label_column = load_dataset(cli)
    feature_dim = 2 * encoder_args.d_input
    features = torch.empty((len(geoms), feature_dim), dtype=torch.float32)
    encoder = Poly2Vec(args=encoder_args, device="cpu")
    encoder.eval()

    print(f"[cache] precomputing Fourier features -> {cache_path}", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for idx, geom in enumerate(tqdm(geoms, desc="fourier-cache", unit="geom")):
            if kind == "polygons":
                features[idx] = encode_polygon_feature(encoder.ft_encoder, geom, feature_dim)
            elif kind == "polylines":
                features[idx] = encode_polyline_feature(encoder.ft_encoder, geom, feature_dim)
            else:
                raise ValueError(kind)
    cache_seconds = time.perf_counter() - t0
    bundle = {
        "dataset": cli.dataset,
        "kind": kind,
        "n_freqs": cli.n_freqs,
        "limit": int(cli.limit or 0),
        "label_column": label_column,
        "feature_dim": feature_dim,
        "num_entities": int(features.shape[0]),
        "num_classes": int(len(classes)),
        "classes": classes,
        "features": features.contiguous(),
        "labels": labels.contiguous(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    print(f"[cache] saved {cache_path} ({cache_seconds:.1f}s)", flush=True)
    return bundle, False, cache_seconds, cache_path


def cache_matches(bundle, cli, expected_feature_dim: int) -> bool:
    requested_limit = int(cli.limit or 0)
    cache_limit = int(bundle.get("limit", 0) or 0)
    if bundle.get("dataset") != cli.dataset:
        return False
    if bundle.get("n_freqs") != cli.n_freqs:
        return False
    if bundle.get("feature_dim") != expected_feature_dim:
        return False
    if requested_limit == 0 and cache_limit != 0:
        return False
    if requested_limit > 0 and bundle.get("num_entities", 0) < requested_limit:
        return False
    if requested_limit > 0 and cache_limit > 0 and cache_limit < requested_limit:
        return False
    return True


def subset_feature_bundle(bundle, limit: int):
    if not limit or bundle["features"].shape[0] <= limit:
        return bundle
    subset = dict(bundle)
    subset["features"] = bundle["features"][:limit].contiguous()
    subset["labels"] = bundle["labels"][:limit].contiguous()
    subset["num_entities"] = int(limit)
    return subset


def load_or_create_feature_cache(cli, encoder_args: SimpleNamespace):
    cache_path = Path(cli.feature_cache_path) if cli.feature_cache_path else default_feature_cache_path(cli)
    if cache_path.exists():
        print(f"[cache] loading {cache_path}", flush=True)
        bundle = torch.load(cache_path, map_location="cpu")
        if cache_matches(bundle, cli, 2 * encoder_args.d_input):
            return subset_feature_bundle(bundle, int(cli.limit or 0)), True, 0.0, cache_path
        print(f"[cache] ignoring incompatible cache at {cache_path}; rebuilding", flush=True)
    return build_feature_cache(cli, encoder_args, cache_path)


def make_loaders(features: torch.Tensor, labels: torch.Tensor, batch_size: int, seed: int, pin_memory: bool):
    base_ds = TensorDataset(features, labels)
    train_idx, val_idx, test_idx = split_indices(len(base_ds), seed)
    train_ds = Subset(base_ds, train_idx.tolist())
    val_ds = Subset(base_ds, val_idx.tolist())
    test_ds = Subset(base_ds, test_idx.tolist())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory),
    )


class CachedShapeClassifier(nn.Module):
    def __init__(self, args: SimpleNamespace, num_classes: int, device: str):
        super().__init__()
        self.encoder = Poly2Vec(args=args, device="cpu")
        self.encoder.nn.to(device)
        self.encoder.param_mag.to(device)
        self.encoder.param_phase.to(device)
        self.head = nn.Sequential(
            nn.Linear(args.d_out, args.d_out),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_out, num_classes),
        ).to(device)
        self.device = device
        self.d_input = args.d_input

    def forward(self, features: torch.Tensor):
        mag = self.encoder.param_mag(features[:, : self.d_input])
        phase = self.encoder.param_phase(features[:, self.d_input :])
        emb = self.encoder.nn(torch.cat([mag, phase], dim=1))
        return self.head(emb)


def run_epoch(model, loader, optimizer, criterion):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    for features, labels in loader:
        features = features.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(features)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total += labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    return total_loss / max(total, 1), correct / max(total, 1)


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-freqs", type=int, default=10)
    parser.add_argument("--d-hid", type=int, default=100)
    parser.add_argument("--d-out", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional dataset cap for smoke tests.")
    parser.add_argument("--device", default=None, help="Training device: cuda | mps | cpu")
    parser.add_argument("--head-device", default=None, help="Deprecated alias for --device")
    parser.add_argument("--encoder-device", default="cpu", help="Cache precompute device; CPU is recommended.")
    parser.add_argument("--feature-cache-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    cli = parse_cli()
    train_device = resolve_device(cli.head_device or cli.device)
    if cli.encoder_device != "cpu":
        print("[device] forcing Fourier feature precompute onto cpu because triangulation is CPU-bound", flush=True)
    print(f"[device] train={train_device} cache_precompute=cpu", flush=True)

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    encoder_args = build_args(cli, train_device)
    bundle, cache_hit, cache_seconds, cache_path = load_or_create_feature_cache(cli, encoder_args)
    features = bundle["features"].to(torch.float32)
    labels = bundle["labels"].to(torch.long)
    kind = bundle["kind"]
    num_classes = int(bundle["num_classes"])

    print(
        f"[cache] hit={cache_hit} N={features.shape[0]} feature_dim={features.shape[1]} kind={kind}",
        flush=True,
    )

    pin_memory = train_device == "cuda"
    train_loader, val_loader, test_loader = make_loaders(
        features, labels, batch_size=cli.batch_size, seed=cli.seed, pin_memory=pin_memory
    )

    model = CachedShapeClassifier(encoder_args, num_classes=num_classes, device=train_device)
    nparams = sum(param.numel() for param in model.parameters())
    optimizer = torch.optim.Adam(
        list(model.encoder.nn.parameters())
        + list(model.encoder.param_mag.parameters())
        + list(model.encoder.param_phase.parameters())
        + list(model.head.parameters()),
        lr=cli.lr,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val = -1.0
    for epoch in range(cli.epochs):
        synchronize_device(train_device)
        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion)
        synchronize_device(train_device)
        epoch_seconds = time.perf_counter() - t0
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_seconds": epoch_seconds,
            }
        )
        best_val = max(best_val, val_acc)
        print(
            f"epoch {epoch + 1:02d}/{cli.epochs:02d} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | {epoch_seconds:.2f}s",
            flush=True,
        )

    test_loss, test_acc = run_epoch(model, test_loader, None, criterion)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}", flush=True)

    summary = {
        "dataset": cli.dataset,
        "kind": kind,
        "num_entities": int(features.shape[0]),
        "num_classes": num_classes,
        "epochs_run": cli.epochs,
        "best_val_accuracy": best_val,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "params": int(nparams),
        "train_device": train_device,
        "feature_cache_path": str(cache_path),
        "feature_cache_hit": cache_hit,
        "feature_cache_seconds": cache_seconds,
        "feature_dim": int(features.shape[1]),
        "encoder_args": vars(encoder_args),
        "history": history,
    }
    out_dir = Path(cli.output_dir or HERE / f"results/{cli.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"[save] {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
