# moirais.fn — function file (hadesllm/moirais)
"""Patience is bitter, but its fruit is sweet. — Aristotle"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
    radius: float = _EARTH_RADIUS_KM,
) -> DescriptiveResult:
    r"""Compute great-circle distance using the Haversine formula.

    .. math::

        d = 2r \arcsin\left(\sqrt{\sin^2\frac{\Delta\phi}{2}
            + \cos\phi_1 \cos\phi_2 \sin^2\frac{\Delta\lambda}{2}}\right)

    Parameters
    ----------
    lat1, lon1 : float or ndarray
        Latitude/longitude of point(s) A in decimal degrees.
    lat2, lon2 : float or ndarray
        Latitude/longitude of point(s) B in decimal degrees.
    radius : float
        Sphere radius in km (default: Earth = 6371 km).

    Returns
    -------
    DescriptiveResult
        name='Haversine Distance', value=distance in km (scalar or mean),
        extra has 'distances' (ndarray if vectorized), 'radius', 'unit'.

    References
    ----------
    Sinnott, R.W. (1984). Virtues of the Haversine. *Sky and
    Telescope*, 68(2), 159.
    """
    lat1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=np.float64))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    distances = radius * c

    val = float(np.mean(distances)) if distances.ndim > 0 else float(distances)

    return DescriptiveResult(
        name="Haversine Distance",
        value=val,
        extra={
            "distances": distances,
            "radius": radius,
            "unit": "km",
        },
    )


havsn = haversine_distance


def cheatsheet() -> str:
    return "haversine_distance({}) -> Haversine great-circle distance. 'Time discovers truth. — Seneca'
