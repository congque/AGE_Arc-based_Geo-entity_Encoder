"""SketchEmbedNet-style baseline on ArcSet shape-classification (few-shot).

Wang et al. (NeurIPS 2020) propose a Conv4 image encoder learned via
sketch-imitation (drawer). Their official codebase is built on TF 2.1 with
heavy custom config/CLI scaffolding and assumes their internal data formats.
For the few-shot Omniglot-stroke / QuickDraw experiments we therefore implement
a *faithful re-implementation* of the encoder architecture (Conv4 backbone with
GlobalAveragePooling, mirroring `models/vae_enc_block.py` in the upstream
repo) and rasterise our gpkg MultiLineStrings to 28x28 binary images, then
train end-to-end via Prototypical Networks. This matches the few-shot
classification protocol used by both the original paper (Sec. 4.2) and our
ArcSet evaluation (`model_edges/testfs.py`).

Why re-implement instead of importing upstream:
  * upstream pinned to tensorflow==2.1, tensorflow-probability==0.8 which
    are EOL and incompatible with current CUDA wheels;
  * upstream's `prepare_data.py` / `run_*.py` need their own caches,
    config files, and slack/horovod glue that adds zero signal here;
  * the encoder architecture is well documented; what matters for the
    benchmark is using the same backbone (Conv4 + GAP).

Rasterisation: each MultiLineString is bbox-normalised to [0, 27] and rendered
via PIL's `ImageDraw` with a 1-pixel stroke; the result is a binary mask.

Usage:
    python run_arcset_dataset.py --dataset single_omniglot --n-way 5 --k-shot 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Force TF to allocate GPU memory growth-style and tame cuBLAS verbosity.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import geopandas as gpd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

DATASETS = {
    "single_omniglot": {
        "path": "data/single_omniglot/omniglot.gpkg",
        "label_col": "label",
    },
    "single_quickdraw": {
        "path": "data/single_quickdraw/quickdraw.gpkg",
        "label_col": "label",
    },
    "single_omniglot_iso": {
        "path": "data/single_omniglot/omniglot_iso.gpkg",
        "label_col": "label",
    },
    "single_quickdraw_iso": {
        "path": "data/single_quickdraw/quickdraw_iso.gpkg",
        "label_col": "label",
    },
}

# Allow GPU memory growth so multiple jobs can share the device.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def _strokes_from_geom(geom):
    """Return a list of (N, 2) float arrays for each stroke in a (Multi)LineString."""
    if isinstance(geom, MultiLineString):
        return [np.asarray(g.coords, dtype=np.float32) for g in geom.geoms]
    if isinstance(geom, LineString):
        return [np.asarray(geom.coords, dtype=np.float32)]
    raise TypeError(f"unexpected geometry: {type(geom).__name__}")


def rasterise_strokes(geom, size=28, pad=2) -> np.ndarray:
    """Render a (Multi)LineString to a binary `size x size` numpy float32 array."""
    strokes = _strokes_from_geom(geom)
    if not strokes:
        return np.zeros((size, size), dtype=np.float32)

    flat = np.concatenate(strokes, axis=0)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)

    # Map into [pad, size - pad - 1]
    drawable = max(1, size - 2 * pad - 1)

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for s in strokes:
        if len(s) < 2:
            continue
        xy = (s - mn) / span * drawable + pad
        # PIL convention: y axis points downward; flip y so the orientation
        # matches the gpkg convention (lower y = bottom of the bbox).
        xy[:, 1] = (size - 1) - xy[:, 1]
        coords = [(float(p[0]), float(p[1])) for p in xy]
        draw.line(coords, fill=255, width=1)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # (size, size)


# ---------------------------------------------------------------------------
# Class-level splits (matches model_edges/testfs.py)
# ---------------------------------------------------------------------------

def class_splits(labels, split_col, seed):
    rng = np.random.default_rng(seed)
    if split_col is not None:
        bg = sorted({l for l, s in zip(labels, split_col) if s == "background"})
        ev = sorted({l for l, s in zip(labels, split_col) if s == "evaluation"})
        if bg and ev:
            bg = list(bg)
            rng.shuffle(bg)
            n_val = max(1, int(len(bg) * 0.1))
            return bg[n_val:], bg[:n_val], ev
    classes = sorted(set(labels.tolist()))
    rng.shuffle(classes)
    n = len(classes)
    n_tr = int(n * 0.7)
    n_va = int(n * 0.15)
    return classes[:n_tr], classes[n_tr:n_tr + n_va], classes[n_tr + n_va:]


def build_index(labels, classes):
    table = {c: [] for c in classes}
    for i, l in enumerate(labels):
        if l in table:
            table[l].append(i)
    return {c: np.asarray(v, dtype=np.int64) for c, v in table.items() if v}


def sample_episode(index_table, classes_pool, n_way, k_shot, n_query, rng):
    available = [c for c in classes_pool if len(index_table[c]) >= k_shot + n_query]
    if len(available) < n_way:
        return None
    chosen = rng.choice(len(available), size=n_way, replace=False)
    classes = [available[i] for i in chosen]
    support, query, query_labels = [], [], []
    for cls_idx, c in enumerate(classes):
        pool = index_table[c]
        pick = rng.choice(len(pool), size=k_shot + n_query, replace=False)
        support.append(pool[pick[:k_shot]])
        query.append(pool[pick[k_shot:]])
        query_labels.extend([cls_idx] * n_query)
    return (
        np.concatenate(support),
        np.concatenate(query),
        np.asarray(query_labels, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Encoder: Conv4 (mirrors SketchEmbedNet vae_enc_block.py BlockConv stack)
# ---------------------------------------------------------------------------

def build_conv4_encoder(image_size: int = 28, embedding_dim: int = 64) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(image_size, image_size, 1))
    x = inp
    for _ in range(4):
        x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)
    return tf.keras.Model(inp, x, name="conv4_encoder")


def proto_logits(z_support, z_query, n_way, k_shot, log_temp, distance="cosine"):
    prototypes = tf.reduce_mean(tf.reshape(z_support, (n_way, k_shot, -1)), axis=1)
    scale = tf.exp(log_temp)
    if distance == "cosine":
        zq = tf.math.l2_normalize(z_query, axis=-1)
        zs = tf.math.l2_normalize(prototypes, axis=-1)
        return scale * tf.matmul(zq, zs, transpose_b=True)
    diff = tf.expand_dims(z_query, 1) - tf.expand_dims(prototypes, 0)
    return -scale * tf.reduce_sum(diff ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Episodic train / eval loop
# ---------------------------------------------------------------------------

def _episode_step(encoder, log_temp, optimizer, sup_imgs, qry_imgs, qry_labels,
                  n_way, k_shot, distance, train: bool):
    if train:
        with tf.GradientTape() as tape:
            z_sup = encoder(sup_imgs, training=True)
            z_qry = encoder(qry_imgs, training=True)
            logits = proto_logits(z_sup, z_qry, n_way, k_shot, log_temp, distance)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    qry_labels, logits, from_logits=True))
        grads = tape.gradient(loss, encoder.trainable_variables + [log_temp])
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables + [log_temp]))
    else:
        z_sup = encoder(sup_imgs, training=False)
        z_qry = encoder(qry_imgs, training=False)
        logits = proto_logits(z_sup, z_qry, n_way, k_shot, log_temp, distance)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                qry_labels, logits, from_logits=True))
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    acc = tf.reduce_mean(tf.cast(preds == qry_labels, tf.float32))
    return float(loss.numpy()), float(acc.numpy())


def run_episodes(encoder, log_temp, raster_array, index_table, classes_pool,
                 n_way, k_shot, n_query, n_episodes, optimizer, rng, distance):
    losses, accs = [], []
    skipped = 0
    while len(losses) < n_episodes and skipped < n_episodes * 4:
        ep = sample_episode(index_table, classes_pool, n_way, k_shot, n_query, rng)
        if ep is None:
            skipped += 1
            continue
        sup_idx, qry_idx, qry_labels = ep
        sup_imgs = raster_array[sup_idx][..., None]
        qry_imgs = raster_array[qry_idx][..., None]
        loss, acc = _episode_step(
            encoder, log_temp, optimizer,
            tf.constant(sup_imgs, dtype=tf.float32),
            tf.constant(qry_imgs, dtype=tf.float32),
            tf.constant(qry_labels, dtype=tf.int64),
            n_way, k_shot, distance, train=optimizer is not None,
        )
        losses.append(loss); accs.append(acc)
    if not accs:
        return float("nan"), float("nan"), float("nan")
    ci = float(np.std(accs) / math.sqrt(len(accs))) * 1.96
    return float(np.mean(losses)), float(np.mean(accs)), ci


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--input", default=None)
    p.add_argument("--label-column", default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--train-episodes", type=int, default=200)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=28)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--init-temp", type=float, default=10.0)
    p.add_argument("--distance", choices=["cosine", "sqeuclidean"], default="cosine")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    path = Path(args.input or (PROJECT_ROOT / spec["path"]))
    label_col = args.label_column or spec["label_col"]
    out_dir = Path(args.output_dir or
        (HERE / f"results/fs_sketchembednet_{args.dataset}_{args.n_way}w_{args.k_shot}s"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {path}", flush=True)
    gdf = gpd.read_file(path)
    if args.limit:
        gdf = gdf.iloc[: args.limit].reset_index(drop=True)
    geoms = gdf.geometry.tolist()
    labels = gdf[label_col].astype(str).to_numpy()
    split_col = gdf["split"].astype(str).to_numpy() if "split" in gdf.columns else None
    print(f"[load] N={len(geoms)} unique_classes={len(set(labels))}", flush=True)

    # Rasterise once.
    t0 = time.time()
    raster = np.zeros((len(geoms), args.image_size, args.image_size), dtype=np.float32)
    for i, g in enumerate(geoms):
        raster[i] = rasterise_strokes(g, size=args.image_size)
        if (i + 1) % 5000 == 0:
            print(f"[rasterise] {i + 1}/{len(geoms)} ({time.time() - t0:.1f}s)", flush=True)
    print(f"[rasterise] done in {time.time() - t0:.1f}s", flush=True)

    train_classes, val_classes, test_classes = class_splits(labels, split_col, args.seed)
    print(f"[classes] train={len(train_classes)} val={len(val_classes)} test={len(test_classes)}",
          flush=True)
    all_classes = train_classes + val_classes + test_classes
    index_table = build_index(labels, all_classes)

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    encoder = build_conv4_encoder(image_size=args.image_size, embedding_dim=args.embedding_dim)
    log_temp = tf.Variable(math.log(args.init_temp), dtype=tf.float32, trainable=True,
                           name="proto_log_temp")
    optimizer = tf.keras.optimizers.Adam(args.lr)
    nparams = int(sum(np.prod(v.shape) for v in encoder.trainable_variables))
    print(f"[model] sketchembednet conv4 params={nparams:,}", flush=True)

    best_val = -1.0
    best_epoch = 0
    best_weights = None
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_episodes(
            encoder, log_temp, raster, index_table, train_classes,
            args.n_way, args.k_shot, args.n_query, args.train_episodes,
            optimizer, rng, args.distance,
        )
        va_loss, va_acc, va_ci = run_episodes(
            encoder, log_temp, raster, index_table, val_classes,
            args.n_way, args.k_shot, args.n_query, args.val_episodes,
            None, rng, args.distance,
        )
        dt = time.time() - t0
        cur_temp = float(np.exp(log_temp.numpy()))
        print(f"epoch {epoch:02d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} "
              f"| val acc={va_acc:.4f} ±{va_ci:.4f} | temp={cur_temp:.3f} | {dt:.1f}s",
              flush=True)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "val_ci95": va_ci,
            "elapsed_s": dt,
        })
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            best_weights = encoder.get_weights()
            best_log_temp = float(log_temp.numpy())

    if best_weights is not None:
        encoder.set_weights(best_weights)
        log_temp.assign(best_log_temp)

    te_loss, te_acc, te_ci = run_episodes(
        encoder, log_temp, raster, index_table, test_classes,
        args.n_way, args.k_shot, args.n_query, args.test_episodes,
        None, rng, args.distance,
    )
    print(f"[test] acc={te_acc:.4f} ±{te_ci:.4f} loss={te_loss:.4f}", flush=True)

    summary = {
        "dataset": args.dataset,
        "model": "sketchembednet",
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "n_query": args.n_query,
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": te_acc,
        "test_ci95": te_ci,
        "test_loss": te_loss,
        "params": nparams,
        "config": {
            "image_size": args.image_size,
            "embedding_dim": args.embedding_dim,
            "distance": args.distance,
            "lr": args.lr,
            "epochs": args.epochs,
            "train_episodes": args.train_episodes,
            "val_episodes": args.val_episodes,
            "test_episodes": args.test_episodes,
            "init_temp": args.init_temp,
            "final_temp": float(np.exp(log_temp.numpy())),
        },
        "history": history,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
