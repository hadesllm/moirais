"""Spatial distance buffer zones. 'Stay on target!' -- Gold Five"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_EARTH_RADIUS_KM = 6371.0


def spatial_buffer(
    lat: np.ndarray,
    lon: np.ndarray,
    radius_km: float,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> DescriptiveResult:
    """Identify points within a distance buffer zone.

    Computes Haversine distances from a center point (or centroid)
    and classifies each point as inside or outside the buffer radius.

    Parameters
    ----------
    lat : ndarray, shape (n_points,)
        Latitudes in decimal degrees.
    lon : ndarray, shape (n_points,)
        Longitudes in decimal degrees.
    radius_km : float
        Buffer radius in kilometers.
    center_lat : float or None
        Center latitude. If None, uses centroid of points.
    center_lon : float or None
        Center longitude. If None, uses centroid of points.

    Returns
    -------
    DescriptiveResult
        name='Spatial Buffer', value=number of points inside buffer,
        extra has 'inside_mask' (bool ndarray), 'distances' (km),
        'center_lat', 'center_lon', 'radius_km'.

    References
    ----------
    de Smith, M.J., Goodchild, M.F. & Longley, P.A. (2018).
    *Geospatial Analysis* (6th ed.). Ch. 4: Distance Operations.
    """
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()

    if center_lat is None:
        center_lat = float(np.mean(lat))
    if center_lon is None:
        center_lon = float(np.mean(lon))

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    clat_r = np.deg2rad(center_lat)
    clon_r = np.deg2rad(center_lon)

    dlat = lat_r - clat_r
    dlon = lon_r - clon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(clat_r) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
    distances = 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    inside = distances <= radius_km
    n_inside = int(np.sum(inside))

    return DescriptiveResult(
        name="Spatial Buffer",
        value=n_inside,
        extra={
            "inside_mask": inside,
            "distances": distances,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km,
            "n_points": len(lat),
            "n_inside": n_inside,
        },
    )


smpbf = spatial_buffer


def cheatsheet() -> str:
    return "spatial_buffer({}) -> Spatial distance buffer zones. 'Stay on target!' -- Gold Fiv"
