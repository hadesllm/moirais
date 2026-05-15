# SPDX-License-Identifier: AGPL-3.0-or-later
"""MRM-framework analyses on Toronto Police Service (TPS) open data.

Python parity for `mrm_tps_*` (see `r-package/morie/R/mrm_tps.R`).

Functions:
    mrm_tps_levy_scaling: Hill-MLE Pareto exponent of inter-incident
        step lengths on lat/long event streams.
    mrm_tps_moran_clustering: gridded Moran's I + DBSCAN summary on
        WGS84 event coordinates.
    mrm_tps_neighbourhood_recurrence_km: per-neighbourhood inter-event
        gap summary.
    mrm_tps_load_hawkes_refit: convenience loader for the per-category
        Hawkes (Markovian + Weibull/sin) manifest.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


__all__ = [
    "mrm_tps_levy_scaling",
    "mrm_tps_moran_clustering",
    "mrm_tps_neighbourhood_recurrence_km",
    "mrm_tps_load_hawkes_refit",
]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


@dataclass
class LevyResult:
    n_events: int
    n_steps_tail: int
    min_step_km: float
    hill_alpha: float


def mrm_tps_levy_scaling(
    data: pd.DataFrame,
    *,
    date_col: str = "OCC_DATE",
    lat_col: str = "LAT_WGS84",
    lon_col: str = "LONG_WGS84",
    min_step_km: float = 0.5,
    x_min: Optional[float] = None,
) -> LevyResult:
    """Hill-MLE exponent on TPS inter-incident step lengths (km)."""
    if x_min is None:
        x_min = min_step_km
    df = data[[date_col, lat_col, lon_col]].dropna().copy()
    df = df.sort_values(date_col)
    lat = df[lat_col].values
    lon = df[lon_col].values
    n = lat.size
    if n < 2:
        return LevyResult(n, 0, min_step_km, float("nan"))
    step = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    tail = step[step >= min_step_km]
    alpha = 1.0 + tail.size / np.log(tail / x_min).sum() if tail.size >= 2 else float("nan")
    return LevyResult(int(n), int(tail.size), float(min_step_km), round(float(alpha), 4))


@dataclass
class MoranClusteringResult:
    morans_I: float
    morans_z: float
    dbscan_n_clusters: int
    dbscan_n_noise: int
    dbscan_largest: int


def mrm_tps_moran_clustering(
    data: pd.DataFrame,
    *,
    lat_col: str = "LAT_WGS84",
    lon_col: str = "LONG_WGS84",
    grid_resolution: int = 40,
    dbscan_eps: float = 0.3,
    dbscan_minpts: int = 5,
) -> MoranClusteringResult:
    """Gridded Moran's I + DBSCAN summary on TPS WGS84 events."""
    df = data[[lat_col, lon_col]].dropna()
    lat = df[lat_col].values
    lon = df[lon_col].values
    n = lat.size
    if n < 10:
        return MoranClusteringResult(float("nan"), float("nan"), 0, 0, 0)

    lat_b = np.linspace(lat.min(), lat.max(), grid_resolution + 1)
    lon_b = np.linspace(lon.min(), lon.max(), grid_resolution + 1)
    i_idx = np.clip(np.digitize(lat, lat_b) - 1, 0, grid_resolution - 1)
    j_idx = np.clip(np.digitize(lon, lon_b) - 1, 0, grid_resolution - 1)
    counts = np.zeros((grid_resolution, grid_resolution), dtype=int)
    np.add.at(counts, (i_idx, j_idx), 1)

    z = counts - counts.mean()
    N = z.size
    num = 0.0
    W = 0
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            if i + 1 < grid_resolution:
                num += z[i, j] * z[i + 1, j]
                W += 1
            if j + 1 < grid_resolution:
                num += z[i, j] * z[i, j + 1]
                W += 1
    num *= 2.0
    W *= 2
    if W == 0 or (z**2).sum() == 0:
        morans_I = float("nan"); morans_z = float("nan")
    else:
        morans_I = float((N / W) * num / (z**2).sum())
        EI = -1.0 / (N - 1)
        var_I = 2.0 / (N - 1) ** 2
        morans_z = (morans_I - EI) / math.sqrt(var_I)

    n_clusters = 0; n_noise = 0; largest = 0
    try:
        from sklearn.cluster import DBSCAN
        pts = np.column_stack([lat * 111.0, lon * 111.0 * math.cos(np.deg2rad(lat.mean()))])
        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minpts).fit_predict(pts)
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        n_noise = int((labels == -1).sum())
        if n_clusters > 0:
            largest = int(pd.Series(labels[labels != -1]).value_counts().max())
    except Exception:
        pass

    return MoranClusteringResult(
        round(morans_I, 6), round(morans_z, 2),
        n_clusters, n_noise, largest,
    )


def mrm_tps_neighbourhood_recurrence_km(
    data: pd.DataFrame,
    *,
    date_col: str = "OCC_DATE",
    hood_col: str = "HOOD_158",
    min_gap_days: float = 0.0,
) -> pd.DataFrame:
    """Per-neighbourhood inter-event gap summary on TPS events."""
    df = data[[date_col, hood_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna().sort_values([hood_col, date_col])
    rows = []
    for hood, sub in df.groupby(hood_col):
        if len(sub) < 2:
            continue
        gaps = sub[date_col].diff().dt.days.dropna().values
        gaps = gaps[gaps >= min_gap_days]
        if gaps.size == 0:
            continue
        rows.append({
            "hood": hood, "n_events": int(len(sub)), "n_gaps": int(gaps.size),
            "mean_gap_days": round(float(gaps.mean()), 2),
            "median_gap_days": float(np.median(gaps)),
            "p25_gap_days": float(np.quantile(gaps, 0.25)),
            "p75_gap_days": float(np.quantile(gaps, 0.75)),
        })
    return pd.DataFrame(rows)


def mrm_tps_load_hawkes_refit(manifest_path: str | Path) -> pd.DataFrame:
    """Load the precomputed per-category Hawkes refit manifest."""
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    d = json.loads(p.read_text())
    rows = []
    for cat, e in d.items():
        rows.append({
            "category": cat,
            "n_fitted": e.get("n_fitted"),
            "T_days": round(e.get("T_days", float("nan")), 1),
            "aic_mark": round(e["markovian"]["aic"], 1),
            "kappa_mark": round(e["markovian"]["branching_ratio"], 3),
            "ks_p_mark": round(e["markovian"]["ks_pvalue"], 3),
            "aic_nm": round(e["weibull_sin"]["aic"], 1),
            "eta_nm": round(e["weibull_sin"]["branching_ratio"], 3),
            "ks_p_nm": round(e["weibull_sin"]["ks_pvalue"], 3),
            "delta_aic": round(e["delta_aic"], 1),
        })
    return pd.DataFrame(rows)
