"""Universal kriging variance."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def universal_kriging_variance(
    coords: np.ndarray,
    target: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    trend_X: np.ndarray | None = None,
) -> SpatialResult:
    r"""Compute universal kriging variance.

    Parameters
    ----------
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    cov_model : str
        Covariance model name.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    trend_X : np.ndarray, optional
        Trend design matrix, shape ``(n, p)``. Defaults to ``[1, x, y]``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the kriging variance.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(coords)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _cov(h):
        if cov_model == "gaussian":
            return sill * np.exp(-((h / rng) ** 2)) + nug * (h == 0)
        return sill * np.exp(-h / rng) + nug * (h == 0)

    if trend_X is None:
        trend_X = np.column_stack([np.ones(n), coords])
    f0 = np.array([1.0, target[0], target[1]])

    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    C = _cov(dist_obs)
    C_inv = np.linalg.inv(C)

    d0 = np.sqrt(((coords - target) ** 2).sum(-1))
    c0 = _cov(d0)
    w = C_inv @ c0

    delta = f0 - trend_X.T @ w
    M = trend_X.T @ C_inv @ trend_X
    var = sill + nug - c0 @ w + delta @ np.linalg.solve(M, delta)

    return SpatialResult(
        name="universal_kriging_variance",
        statistic=float(max(var, 0.0)),
        p_value=None,
        extra={"trend_correction": float(delta @ np.linalg.solve(M, delta))},
    )


sgukv = universal_kriging_variance


def cheatsheet() -> str:
    return "universal_kriging_variance({}) -> Universal kriging variance."
