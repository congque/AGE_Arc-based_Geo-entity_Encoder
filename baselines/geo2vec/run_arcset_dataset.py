"""Geo2Vec adapter for ArcSet with cached SDF samples and GPU training."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset, TensorDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from models import MP_Sampling  # noqa: E402
from models.Geo2Vec import Geo2Vec_Model, SDFLoss  # noqa: E402
from utils.data_loader import preprocessing_list  # noqa: E402

DATASETS = {
    "single_buildings": {
        "path": PROJECT_ROOT / "data" / "single_buildings" / "ShapeClassification.gpkg",
        "label_col": "label",
        "expected_geom": "Polygon",
    },
    "single_mnist": {
        "path": PROJECT_ROOT / "data" / "single_mnist" / "mnist_scaled_normalized.gpkg",
        "label_col": "label",
        "expected_geom": "Polygon",
    },
    "single_omniglot": {
        "path": PROJECT_ROOT / "data" / "single_omniglot" / "omniglot.gpkg",
        "label_col": "label",
        "expected_geom": "MultiLineString",
    },
    "single_quickdraw": {
        "path": PROJECT_ROOT / "data" / "single_quickdraw" / "quickdraw.gpkg",
        "label_col": "label",
        "expected_geom": "MultiLineString",
    },
    # Per-entity isotropic-normalized variants (centroid + max(w,h)/2 -> [-1,1]).
    "single_buildings_iso": {
        "path": PROJECT_ROOT / "data" / "single_buildings" / "ShapeClassification_iso.gpkg",
        "label_col": "label",
        "expected_geom": "Polygon",
    },
    "single_mnist_iso": {
        "path": PROJECT_ROOT / "data" / "single_mnist" / "mnist_iso.gpkg",
        "label_col": "label",
        "expected_geom": "Polygon",
    },
    "single_omniglot_iso": {
        "path": PROJECT_ROOT / "data" / "single_omniglot" / "omniglot_iso.gpkg",
        "label_col": "label",
        "expected_geom": "MultiLineString",
    },
    "single_quickdraw_iso": {
        "path": PROJECT_ROOT / "data" / "single_quickdraw" / "quickdraw_iso.gpkg",
        "label_col": "label",
        "expected_geom": "MultiLineString",
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


def load_arcset_gpkg(path: Path, label_col: str | None, max_entities: int):
    print(f"[load] reading {path}", flush=True)
    gdf = gpd.read_file(path)
    label_col = _detect_label_column(gdf, label_col)
    if max_entities > 0:
        gdf = gdf.iloc[:max_entities].reset_index(drop=True)
    geoms = list(gdf.geometry.values)
    raw_labels = gdf[label_col].astype(str).tolist()
    classes = sorted(set(raw_labels))
    cls2idx = {label: idx for idx, label in enumerate(classes)}
    y = torch.tensor([cls2idx[label] for label in raw_labels], dtype=torch.long)
    print(
        f"[load] N={len(geoms)} classes={len(classes)} label_column={label_col} "
        f"first_geom_type={geoms[0].geom_type}",
        flush=True,
    )
    return geoms, y, classes, label_col


def resolve_device(name: str) -> str:
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if name in {"cuda", "mps"}:
        print(f"[device] {name} requested but unavailable; falling back to cpu", flush=True)
    return "cpu"


def synchronize_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for cls in range(n_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


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


def train_classifier(
    emb: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    device: str,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float = 1e-3,
    hidden: int = 256,
):
    tr_idx, va_idx, te_idx = split_indices(len(y), seed=seed)
    base_ds = TensorDataset(emb.to(torch.float32), y.to(torch.long))
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        Subset(base_ds, tr_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(base_ds, va_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(base_ds, te_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = MLP(emb.shape[1], hidden, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = torch.nn.CrossEntropyLoss()

    def _eval(loader):
        model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                preds = model(xb).argmax(dim=-1)
                preds_all.append(preds.cpu())
                labels_all.append(yb.cpu())
        preds_tensor = torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long)
        labels_tensor = torch.cat(labels_all) if labels_all else torch.empty(0, dtype=torch.long)
        if labels_tensor.numel() == 0:
            return 0.0, 0.0
        acc = (preds_tensor == labels_tensor).float().mean().item()
        f1 = macro_f1_score(labels_tensor.numpy(), preds_tensor.numpy(), n_classes)
        return acc, f1

    history = []
    for ep in range(epochs):
        model.train()
        synchronize_device(device)
        t0 = time.perf_counter()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * yb.shape[0]
            total_seen += yb.shape[0]
        synchronize_device(device)
        epoch_seconds = time.perf_counter() - t0
        train_acc, train_f1 = _eval(DataLoader(Subset(base_ds, tr_idx[: min(2048, len(tr_idx))].tolist()), batch_size=batch_size))
        val_acc, val_f1 = _eval(val_loader)
        test_acc, test_f1 = _eval(test_loader)
        history.append(
            {
                "epoch": ep + 1,
                "loss": total_loss / max(total_seen, 1),
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "test_acc": test_acc,
                "test_macro_f1": test_f1,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"[cls] ep {ep + 1}/{epochs} loss={total_loss / max(total_seen, 1):.4f} "
            f"train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f} "
            f"test_f1={test_f1:.4f} ({epoch_seconds:.2f}s)",
            flush=True,
        )
    return history


def default_sample_cache_path(args) -> Path:
    bw_tag = str(args.sample_band_width_shape).replace(".", "p")
    limit_suffix = f"_limit{args.max_entities}" if args.max_entities > 0 else ""
    return HERE / "cache" / args.dataset / (
        f"cache_{args.dataset}_shape_spu{args.samples_perUnit_shape}_"
        f"pts{args.point_sample_shape}_usp{args.uniformed_sample_perUnit_shape}_"
        f"bw{bw_tag}{limit_suffix}.pt"
    )


def flatten_training_samples(training_samples: dict[int, tuple[list, list]], num_entities: int):
    sample_counts = torch.zeros(num_entities, dtype=torch.long)
    for entity_idx in range(num_entities):
        sample_counts[entity_idx] = len(training_samples[entity_idx][0])
    entity_offsets = torch.zeros(num_entities + 1, dtype=torch.long)
    entity_offsets[1:] = torch.cumsum(sample_counts, dim=0)
    total_samples = int(entity_offsets[-1].item())

    entity_ids = torch.empty(total_samples, dtype=torch.long)
    query_xy = torch.empty((total_samples, 2), dtype=torch.float32)
    gt_sdf = torch.empty((total_samples, 1), dtype=torch.float32)

    for entity_idx in range(num_entities):
        start = int(entity_offsets[entity_idx].item())
        end = int(entity_offsets[entity_idx + 1].item())
        if end <= start:
            continue
        samples, distances = training_samples[entity_idx]
        entity_ids[start:end] = entity_idx
        query_xy[start:end] = torch.tensor(samples, dtype=torch.float32)
        gt_sdf[start:end, 0] = torch.tensor(distances, dtype=torch.float32)

    return entity_ids, query_xy, gt_sdf, entity_offsets, sample_counts


def build_sample_cache(args, cache_path: Path):
    geoms, labels, classes, label_column = load_arcset_gpkg(
        DATASETS[args.dataset]["path"],
        DATASETS[args.dataset]["label_col"],
        args.max_entities,
    )
    torch.set_num_threads(max(args.num_process, 1))
    multiprocessing.set_start_method("spawn", force=True)
    polys_dict_shape, _, _, _, _, _ = preprocessing_list(geoms)

    print(f"[cache] sampling SDF triples -> {cache_path}", flush=True)
    t0 = time.perf_counter()
    training_samples = MP_Sampling.MP_sample(
        polys_dict_shape,
        args.num_process,
        samples_perUnit=args.samples_perUnit_shape,
        point_sample=args.point_sample_shape,
        sample_band_width=args.sample_band_width_shape,
        uniformed_sample_perUnit=args.uniformed_sample_perUnit_shape,
    )
    entity_ids, query_xy, gt_sdf, entity_offsets, sample_counts = flatten_training_samples(
        training_samples,
        num_entities=len(geoms),
    )
    sample_seconds = time.perf_counter() - t0
    total_samples = int(entity_ids.shape[0])
    print(
        f"[cache] sampled {total_samples} points across {len(geoms)} entities "
        f"({total_samples / max(len(geoms), 1):.1f} per entity) in {sample_seconds:.1f}s",
        flush=True,
    )

    bundle = {
        "dataset": args.dataset,
        "limit": int(args.max_entities or 0),
        "label_column": label_column,
        "num_entities": int(len(geoms)),
        "num_classes": int(len(classes)),
        "classes": classes,
        "labels": labels.contiguous(),
        "entity_ids": entity_ids.contiguous(),
        "query_xy": query_xy.contiguous(),
        "gt_sdf": gt_sdf.contiguous(),
        "entity_offsets": entity_offsets.contiguous(),
        "sample_counts": sample_counts.contiguous(),
        "samples_perUnit_shape": int(args.samples_perUnit_shape),
        "point_sample_shape": int(args.point_sample_shape),
        "uniformed_sample_perUnit_shape": int(args.uniformed_sample_perUnit_shape),
        "sample_band_width_shape": float(args.sample_band_width_shape),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)

    del training_samples, polys_dict_shape
    gc.collect()
    return bundle, False, sample_seconds, cache_path


def cache_matches(bundle, args) -> bool:
    requested_limit = int(args.max_entities or 0)
    cache_limit = int(bundle.get("limit", 0) or 0)
    if bundle.get("dataset") != args.dataset:
        return False
    if int(bundle.get("samples_perUnit_shape", -1)) != int(args.samples_perUnit_shape):
        return False
    if int(bundle.get("point_sample_shape", -1)) != int(args.point_sample_shape):
        return False
    if int(bundle.get("uniformed_sample_perUnit_shape", -1)) != int(args.uniformed_sample_perUnit_shape):
        return False
    if float(bundle.get("sample_band_width_shape", -1.0)) != float(args.sample_band_width_shape):
        return False
    if requested_limit == 0 and cache_limit != 0:
        return False
    if requested_limit > 0 and int(bundle.get("num_entities", 0)) < requested_limit:
        return False
    if requested_limit > 0 and cache_limit > 0 and cache_limit < requested_limit:
        return False
    return True


def subset_sample_bundle(bundle, requested_entities: int):
    if requested_entities <= 0 or int(bundle["num_entities"]) <= requested_entities:
        return bundle
    end = int(bundle["entity_offsets"][requested_entities].item())
    subset = dict(bundle)
    subset["num_entities"] = int(requested_entities)
    subset["labels"] = bundle["labels"][:requested_entities].contiguous()
    subset["entity_ids"] = bundle["entity_ids"][:end].contiguous()
    subset["query_xy"] = bundle["query_xy"][:end].contiguous()
    subset["gt_sdf"] = bundle["gt_sdf"][:end].contiguous()
    subset["sample_counts"] = bundle["sample_counts"][:requested_entities].contiguous()
    subset["entity_offsets"] = bundle["entity_offsets"][: requested_entities + 1].clone().contiguous()
    return subset


def load_or_create_sample_cache(args):
    cache_path = Path(args.sample_cache_path) if args.sample_cache_path else default_sample_cache_path(args)
    if cache_path.exists():
        print(f"[cache] loading {cache_path}", flush=True)
        bundle = torch.load(cache_path, map_location="cpu")
        if cache_matches(bundle, args):
            return subset_sample_bundle(bundle, int(args.max_entities or 0)), True, 0.0, cache_path
        print(f"[cache] ignoring incompatible cache at {cache_path}; rebuilding", flush=True)
    return build_sample_cache(args, cache_path)


def make_sample_split_counts(sample_counts: torch.Tensor, training_ratio: float):
    if sample_counts.numel() == 0:
        return sample_counts.clone(), sample_counts.clone()
    train_counts = torch.clamp((sample_counts.to(torch.float32) * training_ratio).round().to(torch.long), min=1)
    train_counts = torch.minimum(train_counts, sample_counts)
    val_counts = sample_counts - train_counts
    needs_val = (sample_counts > 1) & (val_counts == 0)
    train_counts[needs_val] -= 1
    val_counts = sample_counts - train_counts
    return train_counts, val_counts


def iter_sdf_batches(bundle, split: str, batch_size: int, seed: int, epoch: int):
    entity_ids = bundle["entity_ids"]
    query_xy = bundle["query_xy"]
    gt_sdf = bundle["gt_sdf"]
    entity_offsets = bundle["entity_offsets"]
    train_counts = bundle["train_counts"]
    val_counts = bundle["val_counts"]
    num_entities = int(bundle["num_entities"])

    order = np.arange(num_entities)
    if split == "train":
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        counts = train_counts
        starts = entity_offsets[:-1]
    elif split == "val":
        counts = val_counts
        starts = entity_offsets[:-1] + train_counts
    else:
        raise ValueError(split)

    batch_id_parts = []
    batch_xy_parts = []
    batch_sdf_parts = []
    batch_count = 0
    for entity_idx in order:
        count = int(counts[entity_idx].item())
        if count <= 0:
            continue
        start = int(starts[entity_idx].item())
        cursor = 0
        while cursor < count:
            take = min(batch_size - batch_count, count - cursor)
            sl = slice(start + cursor, start + cursor + take)
            batch_id_parts.append(entity_ids[sl])
            batch_xy_parts.append(query_xy[sl])
            batch_sdf_parts.append(gt_sdf[sl])
            batch_count += take
            cursor += take
            if batch_count == batch_size:
                yield torch.cat(batch_id_parts, dim=0), torch.cat(batch_xy_parts, dim=0), torch.cat(batch_sdf_parts, dim=0)
                batch_id_parts = []
                batch_xy_parts = []
                batch_sdf_parts = []
                batch_count = 0
    if batch_count:
        yield torch.cat(batch_id_parts, dim=0), torch.cat(batch_xy_parts, dim=0), torch.cat(batch_sdf_parts, dim=0)


def run_sdf_epoch(model, bundle, optimizer, loss_fn, device: str, batch_size: int, seed: int, epoch: int):
    model.train()
    total_loss = 0.0
    total_seen = 0
    total_batches = 0
    for entity_ids, query_xy, gt_sdf in iter_sdf_batches(bundle, "train", batch_size, seed, epoch):
        entity_ids = entity_ids.to(device, non_blocking=True)
        query_xy = query_xy.to(device, non_blocking=True)
        gt_sdf = gt_sdf.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred_sdf = model(entity_ids, query_xy)
        latent_code = model.poly_embedding_layer(entity_ids)
        loss = loss_fn(pred_sdf, gt_sdf, latent_code)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_seen += entity_ids.shape[0]
        total_batches += 1
    return total_loss / max(total_seen, 1), total_batches


def eval_sdf_epoch(model, bundle, device: str, batch_size: int, seed: int, epoch: int):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_batches = 0
    with torch.no_grad():
        for entity_ids, query_xy, gt_sdf in iter_sdf_batches(bundle, "val", batch_size, seed, epoch):
            entity_ids = entity_ids.to(device, non_blocking=True)
            query_xy = query_xy.to(device, non_blocking=True)
            gt_sdf = gt_sdf.to(device, non_blocking=True)
            pred_sdf = model(entity_ids, query_xy)
            batch_loss = F.l1_loss(pred_sdf, gt_sdf, reduction="mean")
            total_loss += float(batch_loss.item()) * entity_ids.shape[0]
            total_seen += entity_ids.shape[0]
            total_batches += 1
    return total_loss / max(total_seen, 1), total_batches


def train_shape_auto_decoder(args, bundle, device: str):
    num_entities = int(bundle["num_entities"])
    train_counts, val_counts = make_sample_split_counts(bundle["sample_counts"], args.training_ratio_shape)
    bundle = dict(bundle)
    bundle["train_counts"] = train_counts
    bundle["val_counts"] = val_counts

    model = Geo2Vec_Model(
        n_poly=num_entities + 1,
        z_size=args.z_size_shape,
        hidden_size=args.hidden_size_shape,
        num_freqs=args.num_freqs_shape,
        weight_decay=args.weight_decay_shape,
        log_sampling=args.log_sampling_shape,
        polar_fourier=args.polar_fourier_shape,
        num_layers=args.num_layers_shape,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SDFLoss(code_reg_weight=args.code_reg_weight_shape, sum=True)

    best_val_loss = float("inf")
    best_embedding = None
    history = []
    for epoch in range(args.sdf_epochs):
        synchronize_device(device)
        t0 = time.perf_counter()
        train_loss, train_batches = run_sdf_epoch(
            model,
            bundle,
            optimizer,
            loss_fn,
            device,
            args.batch_size,
            args.seed,
            epoch,
        )
        val_loss, val_batches = eval_sdf_epoch(
            model,
            bundle,
            device,
            args.batch_size,
            args.seed,
            epoch,
        )
        synchronize_device(device)
        epoch_seconds = time.perf_counter() - t0
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_batches": train_batches,
                "val_batches": val_batches,
                "train_samples": int(train_counts.sum().item()),
                "val_samples": int(val_counts.sum().item()),
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"[sdf] ep {epoch + 1}/{args.sdf_epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} ({epoch_seconds:.2f}s)",
            flush=True,
        )
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_embedding = model.poly_embedding_layer.weight.detach().cpu().clone()

    if best_embedding is None:
        best_embedding = model.poly_embedding_layer.weight.detach().cpu().clone()
    return best_embedding[:num_entities], history


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--device", default="cuda", help="torch device: cuda | mps | cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Alias that sets both --sdf-epochs and --cls-epochs")
    parser.add_argument("--max-entities", type=int, default=0, help="Optional dataset cap for smoke tests")
    parser.add_argument("--sdf-epochs", type=int, default=80)
    parser.add_argument("--cls-epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-process", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024 * 4)
    parser.add_argument("--samples-perUnit-shape", type=int, default=100)
    parser.add_argument("--point-sample-shape", type=int, default=20)
    parser.add_argument("--sample-band-width-shape", type=float, default=0.1)
    parser.add_argument("--uniformed-sample-perUnit-shape", type=int, default=20)
    parser.add_argument("--num-layers-shape", type=int, default=8)
    parser.add_argument("--z-size-shape", type=int, default=256)
    parser.add_argument("--hidden-size-shape", type=int, default=256)
    parser.add_argument("--num-freqs-shape", type=int, default=8)
    parser.add_argument("--code-reg-weight-shape", type=float, default=1.0)
    parser.add_argument("--weight-decay-shape", type=float, default=0.01)
    parser.add_argument("--training-ratio-shape", type=float, default=0.95)
    parser.add_argument("--polar-fourier-shape", action="store_true", default=False)
    parser.add_argument("--log-sampling-shape", action="store_true", default=True)
    parser.add_argument("--cls-batch-size", type=int, default=256)
    parser.add_argument("--sample-cache-path", default=None)
    parser.add_argument("--save-emb", default="")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_cli()
    if args.epochs is not None:
        args.sdf_epochs = args.epochs
        args.cls_epochs = args.epochs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    print(f"[device] using {device}", flush=True)

    bundle, cache_hit, cache_seconds, cache_path = load_or_create_sample_cache(args)
    labels = bundle["labels"].to(torch.long)
    num_classes = int(bundle["num_classes"])
    total_samples = int(bundle["entity_ids"].shape[0])
    print(
        f"[cache] hit={cache_hit} entities={bundle['num_entities']} samples={total_samples} "
        f"avg_per_entity={total_samples / max(int(bundle['num_entities']), 1):.1f}",
        flush=True,
    )

    shape_embedding, sdf_history = train_shape_auto_decoder(args, bundle, device)
    if args.save_emb:
        np.save(args.save_emb, shape_embedding.numpy())
        print(f"[geo2vec] saved embedding to {args.save_emb}", flush=True)

    classifier_history = train_classifier(
        shape_embedding,
        labels,
        n_classes=num_classes,
        device=device,
        epochs=args.cls_epochs,
        batch_size=args.cls_batch_size,
        seed=args.seed,
    )

    final_cls = classifier_history[-1] if classifier_history else {}
    summary = {
        "dataset": args.dataset,
        "num_entities": int(bundle["num_entities"]),
        "num_classes": num_classes,
        "train_device": device,
        "sample_cache_path": str(cache_path),
        "sample_cache_hit": cache_hit,
        "sample_cache_seconds": cache_seconds,
        "total_sdf_samples": total_samples,
        "avg_sdf_samples_per_entity": total_samples / max(int(bundle["num_entities"]), 1),
        "embedding_dim": int(shape_embedding.shape[1]),
        "sdf_epochs": args.sdf_epochs,
        "cls_epochs": args.cls_epochs,
        "sdf_history": sdf_history,
        "classifier_history": classifier_history,
        "final_test_acc": final_cls.get("test_acc"),
        "final_test_macro_f1": final_cls.get("test_macro_f1"),
    }

    out_dir = Path(args.output_dir or HERE / f"results/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"[save] {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
