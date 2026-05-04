"""Point-centric loader for the arc-vs-point ablation.

Returns each spatial entity as a set of vertices (centered to the entity
centroid). Two output modes:

- mode="raw": (N_pts, 2). Just centered (x, y) per vertex.
- mode="pe":  (N_pts, 2 + 6*K). Same xy + multi-frequency Fourier PE
  (matching the midpoint PE used by ArcSet's geom2set).

This isolates "arc-token vs point-token" with the same decoder. If
ArcSet (arc tokens) > point-pe (same PE, same decoder, but on points),
the gain is from the arc-level structure (length, theta, neighbour
turning) rather than from the input encoding alone.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

try:
    from .load_entities import auto_num_freqs, count_arcs, xy_fourier
except ImportError:
    from load_entities import auto_num_freqs, count_arcs, xy_fourier


def _vertices(geom):
    """Return centered vertex coords (already minus the centroid) per ring/line."""
    pieces = []
    if isinstance(geom, (Polygon, MultiPolygon)):
        c = geom.centroid
        center = np.array([c.x, c.y], dtype=np.float32)
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            pieces.append(np.asarray(poly.exterior.coords[:-1], dtype=np.float32) - center)
            for hole in poly.interiors:
                pieces.append(np.asarray(hole.coords[:-1], dtype=np.float32) - center)
    elif isinstance(geom, (LineString, MultiLineString)):
        c = geom.centroid
        center = np.array([c.x, c.y], dtype=np.float32)
        lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for line in lines:
            pieces.append(np.asarray(line.coords, dtype=np.float32) - center)
    else:
        raise TypeError(f"Unsupported geom: {type(geom)}")

    coords = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 2), dtype=np.float32)
    if len(coords) == 0:
        coords = np.zeros((1, 2), dtype=np.float32)
    return coords


def geom2points(geom, mode="raw", num_freqs=8):
    coords = _vertices(geom)
    if mode == "raw":
        return coords  # (N, 2)
    if mode == "pe":
        return xy_fourier(coords, num_freqs).astype(np.float32, copy=False)
    raise ValueError(f"Unknown mode {mode!r}; use 'raw' or 'pe'.")


def point_feature_dim(mode, num_freqs=8):
    if mode == "raw":
        return 2
    if mode == "pe":
        # xy_fourier returns: xy + sin/cos (2D) + xy + sin/cos (radial)
        # = 2 + 2*num_freqs*2 + 2 + num_freqs*2 = 4 + 6*num_freqs
        return 4 + 6 * num_freqs
    raise ValueError(mode)


def load_gpkg_points(path, label_column="label", mode="raw", num_freqs="auto"):
    gdf = gpd.read_file(path)
    geoms = gdf.geometry.tolist()
    if mode == "pe":
        if num_freqs == "auto" or num_freqs is None:
            avg = float(np.mean([count_arcs(g) for g in geoms]))
            num_freqs = auto_num_freqs(avg)
            print(f"[load_points] avg arcs/entity = {avg:.2f} -> num_freqs = {num_freqs}")
        else:
            num_freqs = int(num_freqs)
    point_sets = [geom2points(g, mode=mode, num_freqs=int(num_freqs) if mode == "pe" else 0)
                  for g in geoms]
    labels = gdf[label_column].to_numpy()
    return point_sets, labels, int(num_freqs) if mode == "pe" else 0
