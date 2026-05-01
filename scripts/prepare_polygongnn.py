"""Convert PolygonGNN building_with_index.pkl into a simple pair pickle.

Input:  list[HeteroData], each carrying two clusters (cluster_id in {0, 1}).
        Each vertex has pos (x, y), polygon_id (ring order), hole_id (0=exterior).
Output: pickle of list[dict] with shapely Polygon objects and the relation label:
        {"poly_a": Polygon, "poly_b": Polygon, "label": str}
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def _build_polygon(pos, polygon_id, hole_id):
    """Reconstruct a Polygon from one cluster's vertices.

    Vertices with hole_id == 0 form the exterior ring; vertices with hole_id > 0
    form interior rings (holes), one ring per distinct hole_id. Vertices within
    each ring are ordered by polygon_id ascending.
    """
    rings = {}
    for i, h in enumerate(hole_id):
        rings.setdefault(int(h), []).append(i)
    for h in rings:
        rings[h].sort(key=lambda i: int(polygon_id[i]))
    if 0 not in rings:
        return None
    exterior = pos[rings[0]]
    if len(exterior) < 3:
        return None
    holes = []
    for h, ids in rings.items():
        if h == 0 or len(ids) < 3:
            continue
        holes.append(pos[ids])
    poly = Polygon(exterior, holes if holes else None)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if not poly.is_valid or poly.is_empty or poly.geom_type != "Polygon":
            return None
    return poly


def convert_one(item):
    pos = np.asarray(item.pos)
    cid = np.asarray(item.cluster_id)
    pid = np.asarray(item.polygon_id)
    hid = np.asarray(item.hole_id)
    label = str(item.y)
    polys = []
    for k in (0, 1):
        m = cid == k
        if not m.any():
            return None
        p = _build_polygon(pos[m], pid[m], hid[m])
        if p is None:
            return None
        polys.append(p)
    return {"poly_a": polys[0], "poly_b": polys[1], "label": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_raw/polygongnn/building_with_index.pkl")
    ap.add_argument("--output", default="data/topo_polygongnn/pairs.pkl")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        items = pickle.load(f)
    print(f"loaded {len(items)} HeteroData")

    out = []
    skipped = 0
    for it in items:
        rec = convert_one(it)
        if rec is None:
            skipped += 1
            continue
        out.append(rec)
    print(f"converted {len(out)} pairs, skipped {skipped}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    labels = sorted({r["label"] for r in out})
    print(f"wrote {out_path}, {len(labels)} unique labels")


if __name__ == "__main__":
    main()
