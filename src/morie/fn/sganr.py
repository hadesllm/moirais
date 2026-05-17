"""Anisotropy ratio and principal direction estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def anisotropy_ratio(Z, coords, n_dirs=8, n_lags=10):
    """Estimate geometric anisotropy ratio and principal direction.

    Fits directional variograms and computes the ratio of maximum to minimum
    effective range.

    .. epigraph:: Give me a place to stand and I will move the earth. -- Archimedes

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    n_dirs : int
        Number of directional bins.
    n_lags : int
        Number of lag classes.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(Z)

    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    angles = np.degrees(np.arctan2(dy, dx)) % 180.0
    tol = 180.0 / (2 * n_dirs)

    max_d = dist.max() / 2.0
    edges = np.linspace(0, max_d, n_lags + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dirs = np.linspace(0, 180, n_dirs, endpoint=False)

    ranges = []
    for d in dirs:
        gamma = np.zeros(n_lags)
        for k in range(n_lags):
            ang_mask = (np.abs(angles - d) <= tol) | (np.abs(angles - d - 180) <= tol)
            lag_mask = (dist > edges[k]) & (dist <= edges[k + 1])
            mask = ang_mask & lag_mask & np.triu(np.ones((n, n), dtype=bool), k=1)
            idx = np.argwhere(mask)
            if len(idx) > 0:
                diffs = Z[idx[:, 0]] - Z[idx[:, 1]]
                gamma[k] = 0.5 * np.mean(diffs**2)

        sill_est = gamma.max() if gamma.max() > 0 else 1.0
        above = np.where(gamma >= 0.95 * sill_est)[0]
        eff_range = float(mids[above[0]]) if len(above) > 0 else float(mids[-1])
        ranges.append(eff_range)

    ranges = np.array(ranges)
    ratio = float(ranges.max() / ranges.min()) if ranges.min() > 0 else 1.0
    principal = float(dirs[np.argmax(ranges)])

    return DescriptiveResult(
        name="anisotropy_ratio",
        value=ratio,
        extra={
            "ratio": ratio,
            "principal_direction": principal,
            "minor_direction": float(dirs[np.argmin(ranges)]),
            "directional_ranges": ranges.tolist(),
            "directions": dirs.tolist(),
        },
    )


sganr = anisotropy_ratio


def cheatsheet() -> str:
    return "anisotropy_ratio({}) -> Anisotropy ratio and principal direction estimation."
