"""Download and convert Google QuickDraw simplified ndjson into a multilinestring gpkg.

Pulls per-category ndjson files from Google Cloud Storage, takes the first
`--samples-per-class` rows, converts each drawing (a list of strokes, each a pair
of x/y arrays) to a shapely MultiLineString, and writes one gpkg with columns:
  - geometry (MultiLineString)
  - label (str: category name)
  - key_id (str)
  - recognized (bool)
  - n_strokes (int)
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString


CATEGORIES_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
NDJSON_URL_TMPL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{cat}.ndjson"


def fetch_categories():
    with urllib.request.urlopen(CATEGORIES_URL, timeout=60) as r:
        text = r.read().decode("utf-8")
    return [c.strip() for c in text.splitlines() if c.strip()]


def stream_ndjson(category, max_rows):
    url = NDJSON_URL_TMPL.format(cat=category.replace(" ", "%20"))
    rows = []
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def drawing_to_geom(drawing):
    strokes = []
    for stroke in drawing:
        if len(stroke) < 2:
            continue
        xs = np.asarray(stroke[0], dtype=np.float64)
        ys = np.asarray(stroke[1], dtype=np.float64)
        if len(xs) < 2 or len(ys) < 2:
            continue
        n = min(len(xs), len(ys))
        pts = np.stack([xs[:n], ys[:n]], axis=1)
        strokes.append(pts)
    if not strokes:
        return None
    pts_all = np.concatenate(strokes, axis=0)
    cx = (pts_all[:, 0].min() + pts_all[:, 0].max()) * 0.5
    cy = (pts_all[:, 1].min() + pts_all[:, 1].max()) * 0.5
    centered = [s - np.array([cx, cy]) for s in strokes]
    scale = float(np.abs(np.concatenate(centered, axis=0)).max())
    if scale > 0:
        centered = [s / scale for s in centered]
    geom = MultiLineString([LineString(s) for s in centered])
    if not geom.is_valid or geom.is_empty:
        return None
    return geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=100,
                    help="number of categories to randomly pick (PDF spec: 100)")
    ap.add_argument("--samples-per-class", type=int, default=10000,
                    help="rows per category (PDF spec: 10000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--categories", default=None,
                    help="optional comma-separated category override (debug)")
    ap.add_argument("--output", default="data/single_quickdraw/quickdraw.gpkg")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.categories:
        chosen = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        cats = fetch_categories()
        print(f"fetched {len(cats)} categories")
        chosen = rng.sample(cats, k=min(args.num_classes, len(cats)))
        chosen.sort()
    print(f"chosen ({len(chosen)}): {chosen[:10]}{'...' if len(chosen) > 10 else ''}")

    rows = []
    for i, cat in enumerate(chosen):
        try:
            data = stream_ndjson(cat, args.samples_per_class)
        except Exception as e:
            print(f"  [{i:03d}/{len(chosen)}] FAIL {cat!r}: {e}")
            continue
        kept = 0
        for d in data:
            geom = drawing_to_geom(d.get("drawing", []))
            if geom is None:
                continue
            rows.append({
                "label": cat,
                "key_id": str(d.get("key_id", "")),
                "recognized": bool(d.get("recognized", False)),
                "n_strokes": len(d.get("drawing", [])),
                "geometry": geom,
            })
            kept += 1
        print(f"  [{i:03d}/{len(chosen)}] {cat!r}: kept {kept}/{len(data)}")

    if not rows:
        raise SystemExit("no rows collected")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GPKG")
    print(f"wrote {out} | {len(gdf)} rows | classes={gdf['label'].nunique()}")


if __name__ == "__main__":
    main()
