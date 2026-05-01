"""Convert Omniglot stroke data to a MultiLineString gpkg.

Walks data_raw/omniglot/strokes_{background,evaluation}/<Alphabet>/<characterNN>/<id>_<rep>.txt.
Each .txt holds one or more strokes separated by lines containing only "START".
Each stroke line is "x,y,t". We build one MultiLineString geometry per .txt instance,
normalize per-instance to fit in [-1, 1] (max-abs scaling on centered coords), and
write a single gpkg with columns:
  - geometry (MultiLineString)
  - label (str: "<alphabet>__<character>")
  - alphabet (str)
  - character (str)
  - instance (int)
  - split (str: "background" | "evaluation")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString


def parse_strokes(txt_path):
    strokes = []
    cur = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "START":
                if len(cur) >= 2:
                    strokes.append(np.asarray(cur, dtype=np.float64))
                cur = []
                continue
            if line == "BREAK":
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            cur.append((x, y))
    if len(cur) >= 2:
        strokes.append(np.asarray(cur, dtype=np.float64))
    return strokes


def normalize(strokes):
    if not strokes:
        return strokes
    pts = np.concatenate(strokes, axis=0)
    cx = (pts[:, 0].min() + pts[:, 0].max()) * 0.5
    cy = (pts[:, 1].min() + pts[:, 1].max()) * 0.5
    centered = [s - np.array([cx, cy]) for s in strokes]
    all_pts = np.concatenate(centered, axis=0)
    scale = float(np.abs(all_pts).max())
    if scale <= 0:
        return centered
    return [s / scale for s in centered]


def collect(omniglot_root: Path, split: str):
    base = omniglot_root / f"strokes_{split}"
    if not base.is_dir():
        return []
    rows = []
    alphabets = sorted([d for d in base.iterdir() if d.is_dir()])
    for alpha_dir in alphabets:
        alphabet = alpha_dir.name
        for char_dir in sorted(alpha_dir.iterdir()):
            if not char_dir.is_dir():
                continue
            character = char_dir.name
            label = f"{alphabet}__{character}"
            for txt in sorted(char_dir.glob("*.txt")):
                strokes = parse_strokes(txt)
                if not strokes:
                    continue
                strokes = normalize(strokes)
                geom = MultiLineString([LineString(s) for s in strokes])
                if not geom.is_valid:
                    continue
                rows.append({
                    "label": label,
                    "alphabet": alphabet,
                    "character": character,
                    "instance": int(txt.stem.split("_")[-1]),
                    "split": split,
                    "geometry": geom,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_raw/omniglot")
    ap.add_argument("--output", default="data/single_omniglot/omniglot.gpkg")
    args = ap.parse_args()

    omniglot_root = Path(args.input)
    rows = []
    for split in ("background", "evaluation"):
        part = collect(omniglot_root, split)
        print(f"[{split}] collected {len(part)} instances")
        rows.extend(part)

    if not rows:
        raise SystemExit("no rows collected; check --input path")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GPKG")
    print(f"wrote {out} | {len(gdf)} rows | classes={gdf['label'].nunique()} | "
          f"bg_classes={gdf[gdf.split=='background']['label'].nunique()} | "
          f"eval_classes={gdf[gdf.split=='evaluation']['label'].nunique()}")


if __name__ == "__main__":
    main()
