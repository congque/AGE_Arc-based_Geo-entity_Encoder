from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon


def freq_bands(num_freqs):
    return (2.0 ** np.linspace(0.0, num_freqs - 1, num_freqs) * np.pi).astype(np.float32)


def xy_fourier(xy, num_freqs):
    bands = freq_bands(num_freqs)
    x_freq = xy[:, None, :] * bands[:, None]
    sin_xy = np.sin(x_freq).reshape(xy.shape[0], -1)
    cos_xy = np.cos(x_freq).reshape(xy.shape[0], -1)

    x = xy[:, 0]
    y = xy[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    r_freq = r[:, None] * bands[None, :]
    sin_r = np.sin(r_freq)
    cos_r = np.cos(r_freq)

    return np.concatenate([xy, sin_xy, cos_xy, xy, sin_r, cos_r], axis=1).astype(np.float32, copy=False)


def scalar_fourier(x, num_freqs):
    bands = freq_bands(num_freqs)
    x_freq = x[:, None] * bands[None, :]
    return np.concatenate([x[:, None], np.sin(x_freq), np.cos(x_freq)], axis=1).astype(np.float32, copy=False)


def edge_feature_dim(xy_num_freqs=8, length_fourier=True, length_num_freqs=None, second_harmonic=True, use_endpoints=False):
    xy_dim = 4 + 6 * xy_num_freqs
    length_dim = 1 + 2 * (length_num_freqs or xy_num_freqs) if length_fourier else 1
    angle_dim = 12 if second_harmonic else 6
    endpoint_dim = xy_dim * 2 if use_endpoints else 0
    return xy_dim + endpoint_dim + length_dim + angle_dim


# 单个 entity -> 单个 edge set
def geom2set(geom, xy_num_freqs=8, length_fourier=True, length_num_freqs=None, second_harmonic=True, use_endpoints=False):
    edges = []

    if isinstance(geom, (Polygon, MultiPolygon)):
        center = geom.centroid
        center = np.array([center.x, center.y], dtype=np.float32)
        polygons = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for polygon in polygons:
            for ring in [polygon.exterior, *polygon.interiors]:
                coords = np.asarray(ring.coords[:-1], dtype=np.float32)
                coords = coords - center
                nxt = np.roll(coords, -1, axis=0)
                vec = nxt - coords
                length = np.linalg.norm(vec, axis=1)
                mid = (coords + nxt) * 0.5
                mid_fourier = xy_fourier(mid, xy_num_freqs)
                theta = np.arctan2(vec[:, 1], vec[:, 0])
                prev_theta = np.roll(theta, 1)
                next_theta = np.roll(theta, -1)
                length_feat = scalar_fourier(length, length_num_freqs or xy_num_freqs) if length_fourier else length[:, None]
                angle_feat = [
                    np.sin(theta),
                    np.cos(theta),
                    np.sin(theta - prev_theta),
                    np.cos(theta - prev_theta),
                    np.sin(next_theta - theta),
                    np.cos(next_theta - theta),
                ]
                if second_harmonic:
                    angle_feat.extend(
                        [
                            np.sin(2 * theta),
                            np.cos(2 * theta),
                            np.sin(2 * (theta - prev_theta)),
                            np.cos(2 * (theta - prev_theta)),
                            np.sin(2 * (next_theta - theta)),
                            np.cos(2 * (next_theta - theta)),
                        ]
                    )
                blocks = [mid_fourier]
                if use_endpoints:
                    blocks.append(xy_fourier(coords, xy_num_freqs))
                    blocks.append(xy_fourier(nxt, xy_num_freqs))
                blocks.append(length_feat)
                blocks.append(np.stack(angle_feat, axis=1))

                edges.append(
                    np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
                )

    elif isinstance(geom, (LineString, MultiLineString)):
        center = geom.centroid
        center = np.array([center.x, center.y], dtype=np.float32)
        lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for line in lines:
            coords = np.asarray(line.coords, dtype=np.float32)
            coords = coords - center
            nxt = coords[1:]
            coords = coords[:-1]
            vec = nxt - coords
            length = np.linalg.norm(vec, axis=1)
            mid = (coords + nxt) * 0.5
            mid_fourier = xy_fourier(mid, xy_num_freqs)
            theta = np.arctan2(vec[:, 1], vec[:, 0])
            prev_delta_sin = np.zeros_like(theta)
            prev_delta_cos = np.zeros_like(theta)
            next_delta_sin = np.zeros_like(theta)
            next_delta_cos = np.zeros_like(theta)
            delta = theta[1:] - theta[:-1]
            prev_delta_sin[1:] = np.sin(delta)
            prev_delta_cos[1:] = np.cos(delta)
            next_delta_sin[:-1] = np.sin(delta)
            next_delta_cos[:-1] = np.cos(delta)
            length_feat = scalar_fourier(length, length_num_freqs or xy_num_freqs) if length_fourier else length[:, None]
            angle_feat = [
                np.sin(theta),
                np.cos(theta),
                prev_delta_sin,
                prev_delta_cos,
                next_delta_sin,
                next_delta_cos,
            ]
            if second_harmonic:
                prev_delta2_sin = np.zeros_like(theta)
                prev_delta2_cos = np.zeros_like(theta)
                next_delta2_sin = np.zeros_like(theta)
                next_delta2_cos = np.zeros_like(theta)
                prev_delta2_sin[1:] = np.sin(2 * delta)
                prev_delta2_cos[1:] = np.cos(2 * delta)
                next_delta2_sin[:-1] = np.sin(2 * delta)
                next_delta2_cos[:-1] = np.cos(2 * delta)
                angle_feat.extend(
                    [
                        np.sin(2 * theta),
                        np.cos(2 * theta),
                        prev_delta2_sin,
                        prev_delta2_cos,
                        next_delta2_sin,
                        next_delta2_cos,
                    ]
                )
            blocks = [mid_fourier]
            if use_endpoints:
                blocks.append(xy_fourier(coords, xy_num_freqs))
                blocks.append(xy_fourier(nxt, xy_num_freqs))
            blocks.append(length_feat)
            blocks.append(np.stack(angle_feat, axis=1))

            edges.append(
                np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
            )

    elif isinstance(geom, (Point, MultiPoint)):
        points = list(geom.geoms) if isinstance(geom, MultiPoint) else [geom]
        for _ in points:
            edges.append(
                np.zeros(
                    (
                        1,
                        edge_feature_dim(
                            xy_num_freqs=xy_num_freqs,
                            length_fourier=length_fourier,
                            length_num_freqs=length_num_freqs,
                            second_harmonic=second_harmonic,
                            use_endpoints=use_endpoints,
                        ),
                    ),
                    dtype=np.float32,
                )
            )

    return np.concatenate(edges, axis=0)


def geoms2sets(geom_list, **kwargs):
    return [geom2set(geom, **kwargs) for geom in geom_list]


def load_gpkg(path, label_column="label", **kwargs):
    gdf = gpd.read_file(path)
    edge_sets = geoms2sets(gdf.geometry.tolist(), **kwargs)
    labels = gdf[label_column].to_numpy()
    return edge_sets, labels
