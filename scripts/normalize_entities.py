"""Per-entity isotropic standardization for ArcSet datasets.

Reads a gpkg, applies per-entity affine: subtract bbox center, divide by
max(width, height) / 2 so the longest side spans [-1, 1] and the shorter
side keeps its true aspect ratio. Writes a new gpkg next to the original.

Usage:
    python scripts/normalize_entities.py \
        --input data/single_buildings/ShapeClassification.gpkg \
        --output data/single_buildings/ShapeClassification_iso.gpkg

Or run the default sweep over all four datasets:
    python scripts/normalize_entities.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.affinity import affine_transform


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGETS = [
    ("data/single_buildings/ShapeClassification.gpkg",
     "data/single_buildings/ShapeClassification_iso.gpkg"),
    ("data/single_mnist/mnist_scaled_normalized.gpkg",
     "data/single_mnist/mnist_iso.gpkg"),
    ("data/single_omniglot/omniglot.gpkg",
     "data/single_omniglot/omniglot_iso.gpkg"),
    ("data/single_quickdraw/quickdraw.gpkg",
     "data/single_quickdraw/quickdraw_iso.gpkg"),
]


def isotropic_normalize(geom):
    """Affine: subtract bbox center, scale by 2 / max(w, h) so longer axis
    spans [-1, 1]. Aspect ratio preserved."""
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    s = max(w, h)
    if s <= 0:
        return geom
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    factor = 2.0 / s
    # shapely affine_transform: [a, b, d, e, xoff, yoff]
    # new_x = a*x + b*y + xoff = factor * (x - cx)
    # new_y = d*x + e*y + yoff = factor * (y - cy)
    return affine_transform(geom, [factor, 0, 0, factor, -factor * cx, -factor * cy])


def normalize_file(src: Path, dst: Path):
    print(f"[norm] reading  {src}")
    gdf = gpd.read_file(src)
    n = len(gdf)
    print(f"[norm] {n} entities; applying per-entity isotropic standardize")
    gdf["geometry"] = gdf.geometry.apply(isotropic_normalize)
    bounds = gdf.total_bounds
    print(f"[norm] post-norm total_bounds: {bounds}")
    sample_bounds = [g.bounds for g in gdf.geometry.iloc[:3]]
    print(f"[norm] first 3 entity bounds: {sample_bounds}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[norm] writing  {dst}")
    gdf.to_file(dst, driver="GPKG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Normalize all 4 default datasets")
    args = parser.parse_args()

    if args.all:
        for src_rel, dst_rel in DEFAULT_TARGETS:
            src = REPO_ROOT / src_rel
            dst = REPO_ROOT / dst_rel
            if not src.exists():
                print(f"[norm] SKIP {src} (missing)")
                continue
            if dst.exists():
                print(f"[norm] SKIP {dst} (exists, delete to redo)")
                continue
            normalize_file(src, dst)
            print()
        return

    if args.input is None or args.output is None:
        parser.error("Provide --input and --output, or use --all")

    normalize_file(args.input, args.output)


if __name__ == "__main__":
    main()
