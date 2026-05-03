"""Sketchformer-style baseline on ArcSet shape-classification (few-shot).

Ribeiro et al. (CVPR 2020) introduced a Transformer that consumes sketches as
sequences of (Δx, Δy, pen-state) triplets (the SketchRNN "stroke-3" /
"stroke-5" format). Their official codebase is built on TF 2.1 with
hand-rolled trainer scaffolding, slack notifiers, and a heavy dataset
pipeline that depends on chunked QuickDraw npz files. We re-implement the
encoder in pure TF2/Keras (multi-head self-attention with positional
encoding, mean-pool to the lowerdim bottleneck) so it loads our gpkg dataset
and trains end-to-end with a Prototypical Networks decoder, matching the
ArcSet few-shot evaluation protocol in `model_edges/testfs.py`.

Why re-implement instead of importing upstream:
  * upstream pinned to tensorflow-gpu==2.1, slackclient, horovod, custom
    `core.models.BaseModel` trainer, and chunked QuickDraw npz pipeline that
    is unrelated to our gpkg datasets;
  * core architectural choices (continuous stroke-5 input, multi-head
    self-attention encoder, lowerdim bottleneck via attention pooling) are
    well documented and trivial to mirror in modern Keras;
  * what matters for the benchmark is that the *encoder family* (Transformer
    on stroke-3) is represented honestly.

Stroke-3 conversion: each MultiLineString is bbox-normalised to [-1, 1],
and consecutive points within a stroke produce (Δx, Δy, 0) triples; the last
point of every stroke (except the final stroke) emits (Δx, Δy, 1) — a
pen-up. Sequences are padded/truncated to `--max-seq-len`.

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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import geopandas as gpd
import numpy as np
import tensorflow as tf
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
}

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Stroke-3 conversion
# ---------------------------------------------------------------------------

def _strokes_from_geom(geom):
    if isinstance(geom, MultiLineString):
        return [np.asarray(g.coords, dtype=np.float32) for g in geom.geoms]
    if isinstance(geom, LineString):
        return [np.asarray(geom.coords, dtype=np.float32)]
    raise TypeError(f"unexpected geometry: {type(geom).__name__}")


def to_stroke3(geom, max_seq_len: int) -> np.ndarray:
    """Convert (Multi)LineString to padded stroke-3 array of shape (max_seq_len, 3).

    Layout: [Δx, Δy, pen_state] where pen_state == 1 marks a pen-up
    (end-of-stroke) and pen_state == 0 means continue. Padding tokens use
    [0, 0, 0]; we additionally write a final pen-up to mark the end of the
    sketch. Coordinates are bbox-normalised to [-1, 1] before being deltad.
    """
    strokes = _strokes_from_geom(geom)
    if not strokes:
        return np.zeros((max_seq_len, 3), dtype=np.float32)

    flat = np.concatenate(strokes, axis=0)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    # Map to [-1, 1]
    normed = [(s - mn) / span * 2.0 - 1.0 for s in strokes]

    out = np.zeros((max_seq_len, 3), dtype=np.float32)
    write = 0
    prev = np.array([0.0, 0.0], dtype=np.float32)
    for s_i, s in enumerate(normed):
        if len(s) < 1 or write >= max_seq_len:
            break
        # delta from previous point in the sketch
        for pt_i, pt in enumerate(s):
            if write >= max_seq_len:
                break
            dx, dy = pt[0] - prev[0], pt[1] - prev[1]
            pen = 1.0 if pt_i == len(s) - 1 and s_i < len(normed) - 1 else 0.0
            out[write] = [dx, dy, pen]
            write += 1
            prev = pt
    if write < max_seq_len:
        out[write] = [0.0, 0.0, 1.0]  # explicit end-of-sketch
    return out


# ---------------------------------------------------------------------------
# Class splits
# ---------------------------------------------------------------------------

def class_splits(labels, split_col, seed):
    rng = np.random.default_rng(seed)
    if split_col is not None:
        bg = sorted({l for l, s in zip(labels, split_col) if s == "background"})
        ev = sorted({l for l, s in zip(labels, split_col) if s == "evaluation"})
        if bg and ev:
            bg = list(bg); rng.shuffle(bg)
            n_val = max(1, int(len(bg) * 0.1))
            return bg[n_val:], bg[:n_val], ev
    classes = sorted(set(labels.tolist()))
    rng.shuffle(classes)
    n = len(classes)
    n_tr = int(n * 0.7); n_va = int(n * 0.15)
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
# Sketchformer-style encoder
# ---------------------------------------------------------------------------

def _positional_encoding(length: int, depth: int) -> tf.Tensor:
    pos = np.arange(length)[:, None]
    i = np.arange(depth)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / np.float32(depth))
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.cast(angle, tf.float32)


class SketchformerEncoder(tf.keras.Model):
    """Sketchformer-style continuous stroke-3 encoder.

    Mirrors the upstream `Encoder` (4 layers, 8 heads, d_model=128, dff=512,
    lowerdim=256) but with mean-pooling over the (masked) sequence axis as the
    bottleneck — equivalent to upstream's SelfAttnV1 pooling collapsed to a
    plain mean once `lowerdim` projection is applied.
    """

    def __init__(self, num_layers=4, d_model=128, num_heads=8, dff=512,
                 lowerdim=256, max_seq_len=200, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.lowerdim = lowerdim
        self.input_proj = tf.keras.layers.Dense(d_model)
        self.pos_encoding = _positional_encoding(max_seq_len, d_model)
        self.attn_layers = []
        self.ffn_layers = []
        self.norm1 = []
        self.norm2 = []
        self.dropouts1 = []
        self.dropouts2 = []
        for _ in range(num_layers):
            self.attn_layers.append(tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate))
            self.ffn_layers.append(tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]))
            self.norm1.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.norm2.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.dropouts1.append(tf.keras.layers.Dropout(dropout_rate))
            self.dropouts2.append(tf.keras.layers.Dropout(dropout_rate))
        self.bottleneck = tf.keras.layers.Dense(lowerdim)

    def call(self, x, training=False):
        # x: (B, T, 3); pen_state == 1 marks end-of-stroke and the trailing
        # padding region is the all-zero block we wrote in to_stroke3 after
        # the final EOS token. We mark those rows as "pad" via a heuristic:
        # all-zero rows that come after the first explicit EOS (pen=1) are
        # padding.
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        # Padding mask: True where all three channels are zero.
        is_zero = tf.reduce_all(tf.equal(x, 0.0), axis=-1)  # (B, T)
        # Cumulative: a position is treated as padding only if every
        # downstream position is also all-zero (so internal accidental zero
        # deltas are kept). Reverse-cumulative-prod over the reversed axis
        # gives 1 from the tail of the sequence backwards while we're still
        # in the all-zero region.
        rev = tf.reverse(tf.cast(is_zero, tf.float32), axis=[1])
        rev_cum = tf.math.cumprod(rev, axis=1)
        pad_mask = tf.cast(tf.reverse(rev_cum, axis=[1]), tf.bool)  # (B, T)
        seq_mask = tf.logical_not(pad_mask)  # True where token is real
        attn_mask = tf.cast(seq_mask, tf.float32)  # (B, T)

        h = self.input_proj(x) + self.pos_encoding[None, :tf.shape(x)[1], :]
        # MultiHeadAttention `attention_mask` is expected to have shape
        # (B, T_q, T_k) or broadcastable; we feed (B, 1, T) which broadcasts
        # over the query axis.
        mask_2d = attn_mask[:, None, :]
        for i in range(self.num_layers):
            attn_out = self.attn_layers[i](
                query=h, value=h, key=h,
                attention_mask=mask_2d, training=training)
            attn_out = self.dropouts1[i](attn_out, training=training)
            h = self.norm1[i](h + attn_out)
            ffn_out = self.ffn_layers[i](h, training=training)
            ffn_out = self.dropouts2[i](ffn_out, training=training)
            h = self.norm2[i](h + ffn_out)

        # Mean-pool over real tokens; if every token is padding, fall back
        # to a mean of zeros (sequence essentially empty).
        mask = tf.cast(seq_mask, tf.float32)[..., None]
        denom = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        pooled = tf.reduce_sum(h * mask, axis=1) / denom
        return self.bottleneck(pooled)


def proto_logits(z_support, z_query, n_way, k_shot, log_temp, distance="cosine"):
    prototypes = tf.reduce_mean(tf.reshape(z_support, (n_way, k_shot, -1)), axis=1)
    scale = tf.exp(log_temp)
    if distance == "cosine":
        zq = tf.math.l2_normalize(z_query, axis=-1)
        zs = tf.math.l2_normalize(prototypes, axis=-1)
        return scale * tf.matmul(zq, zs, transpose_b=True)
    diff = tf.expand_dims(z_query, 1) - tf.expand_dims(prototypes, 0)
    return -scale * tf.reduce_sum(diff ** 2, axis=-1)


def _episode_step(encoder, log_temp, optimizer, sup_seq, qry_seq, qry_labels,
                  n_way, k_shot, distance, train: bool):
    if train:
        with tf.GradientTape() as tape:
            z_sup = encoder(sup_seq, training=True)
            z_qry = encoder(qry_seq, training=True)
            logits = proto_logits(z_sup, z_qry, n_way, k_shot, log_temp, distance)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    qry_labels, logits, from_logits=True))
        grads = tape.gradient(loss, encoder.trainable_variables + [log_temp])
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables + [log_temp]))
    else:
        z_sup = encoder(sup_seq, training=False)
        z_qry = encoder(qry_seq, training=False)
        logits = proto_logits(z_sup, z_qry, n_way, k_shot, log_temp, distance)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                qry_labels, logits, from_logits=True))
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    acc = tf.reduce_mean(tf.cast(preds == qry_labels, tf.float32))
    return float(loss.numpy()), float(acc.numpy())


def run_episodes(encoder, log_temp, seq_array, index_table, classes_pool,
                 n_way, k_shot, n_query, n_episodes, optimizer, rng, distance):
    losses, accs = [], []
    skipped = 0
    while len(losses) < n_episodes and skipped < n_episodes * 4:
        ep = sample_episode(index_table, classes_pool, n_way, k_shot, n_query, rng)
        if ep is None:
            skipped += 1
            continue
        sup_idx, qry_idx, qry_labels = ep
        loss, acc = _episode_step(
            encoder, log_temp, optimizer,
            tf.constant(seq_array[sup_idx], dtype=tf.float32),
            tf.constant(seq_array[qry_idx], dtype=tf.float32),
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
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq-len", type=int, default=200)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dff", type=int, default=512)
    p.add_argument("--lowerdim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--init-temp", type=float, default=10.0)
    p.add_argument("--distance", choices=["cosine", "sqeuclidean"], default="cosine")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    path = Path(args.input or (PROJECT_ROOT / spec["path"]))
    label_col = args.label_column or spec["label_col"]
    out_dir = Path(args.output_dir or
        (HERE / f"results/fs_sketchformer_{args.dataset}_{args.n_way}w_{args.k_shot}s"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {path}", flush=True)
    gdf = gpd.read_file(path)
    if args.limit:
        gdf = gdf.iloc[: args.limit].reset_index(drop=True)
    geoms = gdf.geometry.tolist()
    labels = gdf[label_col].astype(str).to_numpy()
    split_col = gdf["split"].astype(str).to_numpy() if "split" in gdf.columns else None
    print(f"[load] N={len(geoms)} unique_classes={len(set(labels))}", flush=True)

    t0 = time.time()
    seq = np.zeros((len(geoms), args.max_seq_len, 3), dtype=np.float32)
    for i, g in enumerate(geoms):
        seq[i] = to_stroke3(g, max_seq_len=args.max_seq_len)
        if (i + 1) % 5000 == 0:
            print(f"[stroke3] {i + 1}/{len(geoms)} ({time.time() - t0:.1f}s)", flush=True)
    print(f"[stroke3] done in {time.time() - t0:.1f}s", flush=True)

    train_classes, val_classes, test_classes = class_splits(labels, split_col, args.seed)
    print(f"[classes] train={len(train_classes)} val={len(val_classes)} test={len(test_classes)}",
          flush=True)
    all_classes = train_classes + val_classes + test_classes
    index_table = build_index(labels, all_classes)

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    encoder = SketchformerEncoder(
        num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads,
        dff=args.dff, lowerdim=args.lowerdim, max_seq_len=args.max_seq_len,
        dropout_rate=args.dropout)
    # Build by running a dummy batch.
    _ = encoder(tf.zeros((1, args.max_seq_len, 3), dtype=tf.float32), training=False)
    log_temp = tf.Variable(math.log(args.init_temp), dtype=tf.float32, trainable=True,
                           name="proto_log_temp")
    optimizer = tf.keras.optimizers.Adam(args.lr)
    nparams = int(sum(np.prod(v.shape) for v in encoder.trainable_variables))
    print(f"[model] sketchformer params={nparams:,}", flush=True)

    best_val = -1.0
    best_epoch = 0
    best_weights = None
    best_log_temp = float(log_temp.numpy())
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_episodes(
            encoder, log_temp, seq, index_table, train_classes,
            args.n_way, args.k_shot, args.n_query, args.train_episodes,
            optimizer, rng, args.distance,
        )
        va_loss, va_acc, va_ci = run_episodes(
            encoder, log_temp, seq, index_table, val_classes,
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
        encoder, log_temp, seq, index_table, test_classes,
        args.n_way, args.k_shot, args.n_query, args.test_episodes,
        None, rng, args.distance,
    )
    print(f"[test] acc={te_acc:.4f} ±{te_ci:.4f} loss={te_loss:.4f}", flush=True)

    summary = {
        "dataset": args.dataset,
        "model": "sketchformer",
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
            "max_seq_len": args.max_seq_len,
            "num_layers": args.num_layers,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "dff": args.dff,
            "lowerdim": args.lowerdim,
            "dropout": args.dropout,
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
