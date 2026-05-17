"""Isotropy test via directional variogram comparison."""

from __future__ import annotations

from ._containers import DescriptiveResult


def isotropy_test(Z, coords, n_dirs=4, n_lags=10, tol_deg=22.5):
    """Test isotropy by comparing directional semivariograms.

    Computes variograms in *n_dirs* angular sectors and tests whether they
    differ significantly.

    .. epigraph:: It is not what happens to you, but how you react, that matters. -- Epictetus

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    n_dirs : int
        Number of directional sectors.
    n_lags : int
        Number of lag bins.
    tol_deg : float
        Angular tolerance in degrees.

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

    max_d = dist.max() / 2.0
    edges = np.linspace(0, max_d, n_lags + 1)
    dirs = np.linspace(0, 180, n_dirs, endpoint=False)

    dir_varios = {}
    for d in dirs:
        gamma = np.zeros(n_lags)
        for k in range(n_lags):
            ang_mask = np.abs(angles - d) <= tol_deg
            ang_mask |= np.abs(angles - d - 180) <= tol_deg
            lag_mask = (dist > edges[k]) & (dist <= edges[k + 1])
            mask = ang_mask & lag_mask & (np.triu(np.ones((n, n), dtype=bool), k=1))
            idx = np.argwhere(mask)
            if len(idx) > 0:
                diffs = Z[idx[:, 0]] - Z[idx[:, 1]]
                gamma[k] = 0.5 * np.mean(diffs**2)
        dir_varios[float(d)] = gamma

    values = np.array(list(dir_varios.values()))
    if values.shape[0] < 2:
        return DescriptiveResult(
            name="isotropy_test",
            value=0.0,
            extra={"isotropic": True, "directions": dirs.tolist()},
        )

    mean_vario = values.mean(axis=0)
    max_dev = 0.0
    for row in values:
        nz = mean_vario > 0
        if nz.any():
            dev = np.max(np.abs(row[nz] - mean_vario[nz]) / mean_vario[nz])
            max_dev = max(max_dev, dev)

    return DescriptiveResult(
        name="isotropy_test",
        value=float(max_dev),
        extra={
            "max_relative_deviation": float(max_dev),
            "isotropic": max_dev < 0.3,
            "directions": dirs.tolist(),
            "n_dirs": n_dirs,
        },
    )


sgiso = isotropy_test


def cheatsheet() -> str:
    return "isotropy_test({}) -> Isotropy test via directional variogram comparison."
