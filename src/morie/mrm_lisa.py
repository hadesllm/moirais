# SPDX-License-Identifier: AGPL-3.0-or-later
"""LISA (Local Indicators of Spatial Association) + per-year polygon
Moran's I on TPS-style geographic crime data.

Two callables:

1. mrm_tps_lisa(data, year=...) -- Local Moran's I per polygon centroid,
   with 999-permutation Monte-Carlo significance. Returns per-polygon
   I_i, lag z-score, quadrant (HH/HL/LH/LL), and p-value. Computes
   the global Moran's I as a by-product.

2. mrm_tps_polygon_moran_per_year(data, year_cols=...) -- convenience
   wrapper that loops mrm_tps_lisa over a set of per-year count
   columns (e.g. ASSAULT_2014 ... ASSAULT_2024) and returns the
   global Moran's I time series. Used by the empirical paper §7.11
   to document the declining spatial concentration of assault
   2014-2024.

Both functions take a GeoJSON-like polygon frame (or a DataFrame
with lat/lon centroid columns + per-feature counts) and compute
k-nearest-neighbour spatial weights (default k=6).

References:
    Anselin, L. (1995). Local indicators of spatial association -- LISA.
        Geographical Analysis, 27(2), 93-115.
    Anselin, L. (2010). Thirty years of spatial econometrics.
        Papers in Regional Science, 89(1), 3-25.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "LISAResult",
    "mrm_tps_lisa",
    "mrm_tps_polygon_moran_per_year",
]


@dataclass
class LISAResult:
    """Output of mrm_tps_lisa."""

    n_polygons: int
    global_moran_I: float
    permutations: int
    knn_k: int
    table: pd.DataFrame   # per-polygon: id, lat, lon, x, z, lag_z, I_local, quadrant, p_value
    quadrants_all: dict
    quadrants_significant_p05: dict
    n_significant_p05: int


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _knn_weights(lat: np.ndarray, lon: np.ndarray, k: int) -> np.ndarray:
    n = lat.size
    W = np.zeros((n, n))
    for i in range(n):
        d = _haversine_km(lat[i], lon[i], lat, lon)
        nn = np.argsort(d)[1:k + 1]
        W[i, nn] = 1.0 / k
    return W


def mrm_tps_lisa(
    data: pd.DataFrame,
    *,
    count_col: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str | None = None,
    k: int = 6,
    n_permutations: int = 999,
    seed: int = 42,
) -> LISAResult:
    """Local Moran's I + quadrant + significance for a polygon surface.

    Args:
        data: DataFrame with one row per polygon. Must have lat/lon
            centroid columns + a count_col with per-polygon counts.
        count_col: per-polygon count column (e.g. "ASSAULT_2024").
        lat_col, lon_col: WGS84 centroid columns.
        id_col: optional polygon-id column (passed through to output).
        k: k-NN spatial-weights neighbourhood (default 6).
        n_permutations: MC permutations for significance. 999 is the
            spatial-statistics convention.
        seed: RNG seed for reproducibility.

    Returns:
        LISAResult with per-polygon table and quadrant summaries.
    """
    rng = np.random.default_rng(seed)
    df = data[[count_col, lat_col, lon_col] + ([id_col] if id_col else [])].copy()
    df = df.dropna(subset=[count_col, lat_col, lon_col])
    n = len(df)
    if n < 5:
        raise ValueError(f"need >= 5 polygons; got {n}")

    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    x = df[count_col].to_numpy(dtype=float)

    W = _knn_weights(lat, lon, k)
    z = (x - x.mean()) / x.std(ddof=0)
    lag = W @ z
    I_local = z * lag
    I_global = float(I_local.sum() / (z ** 2).sum())

    # Quadrants
    quad = np.empty(n, dtype=object)
    quad[(z > 0) & (lag > 0)] = "HH"
    quad[(z > 0) & (lag <= 0)] = "HL"
    quad[(z <= 0) & (lag > 0)] = "LH"
    quad[(z <= 0) & (lag <= 0)] = "LL"

    # Significance via MC permutation (z fixed, lag permuted)
    p_local = np.zeros(n)
    for _ in range(n_permutations):
        zp = rng.permutation(z)
        lp = W @ zp
        p_local += (np.abs(z * lp) >= np.abs(I_local)).astype(int)
    p_local = (p_local + 1) / (n_permutations + 1)

    out = pd.DataFrame({
        "id":         df[id_col].to_numpy() if id_col else range(n),
        "lat":        lat, "lon": lon, "x": x, "z": z,
        "lag_z":      lag, "I_local": I_local,
        "quadrant":   quad, "p_value": p_local,
        "significant_p05": p_local <= 0.05,
    })

    quads_all = {q: int((quad == q).sum()) for q in ("HH", "HL", "LH", "LL")}
    quads_sig = {q: int(((quad == q) & (p_local <= 0.05)).sum())
                 for q in ("HH", "HL", "LH", "LL")}

    return LISAResult(
        n_polygons=int(n),
        global_moran_I=round(I_global, 4),
        permutations=int(n_permutations),
        knn_k=int(k),
        table=out,
        quadrants_all=quads_all,
        quadrants_significant_p05=quads_sig,
        n_significant_p05=int(sum(quads_sig.values())),
    )


def mrm_tps_polygon_moran_per_year(
    data: pd.DataFrame,
    *,
    year_cols: Sequence[str],
    lat_col: str = "lat",
    lon_col: str = "lon",
    k: int = 6,
    n_permutations: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-year global Moran's I time series across a polygon surface.

    Args:
        data: DataFrame with one row per polygon + one count column
            per year.
        year_cols: list of per-year count columns (in year order).
        lat_col, lon_col: WGS84 centroid columns.
        k: k-NN spatial-weights neighbourhood.
        n_permutations: MC permutations for each year's significance.
        seed: RNG seed.

    Returns:
        DataFrame with one row per year: year, n_events, moran_I,
        global_p_value.
    """
    rows = []
    for c in year_cols:
        # Try to parse year out of the column name
        import re
        m = re.search(r"\d{4}", c)
        year = int(m.group(0)) if m else c

        try:
            res = mrm_tps_lisa(
                data, count_col=c,
                lat_col=lat_col, lon_col=lon_col,
                k=k, n_permutations=n_permutations, seed=seed,
            )
        except ValueError:
            continue

        # Global p via permutation of the whole z surface
        rng = np.random.default_rng(seed)
        n = res.n_polygons
        x = data[c].dropna().to_numpy(dtype=float)
        z = (x - x.mean()) / x.std(ddof=0)
        lat = data[lat_col].dropna().to_numpy(dtype=float)
        lon = data[lon_col].dropna().to_numpy(dtype=float)
        W = _knn_weights(lat, lon, k)
        I_obs = res.global_moran_I
        gt = 0
        for _ in range(n_permutations):
            zp = rng.permutation(z)
            I_perm = (zp * (W @ zp)).sum() / (zp ** 2).sum()
            if abs(I_perm) >= abs(I_obs):
                gt += 1
        p_global = (gt + 1) / (n_permutations + 1)

        rows.append({
            "year": year,
            "n_events": int(x.sum()),
            "moran_I": I_obs,
            "global_p_value": round(p_global, 4),
        })

    return pd.DataFrame(rows)
