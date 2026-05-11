"""Simple kriging variance."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def simple_kriging_variance(
    coords: np.ndarray,
    target: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
) -> SpatialResult:
    r"""Compute simple kriging variance without observations.

    .. math::

        \sigma_{SK}^2(s_0) = C(0) - \mathbf{c}_0^T \mathbf{C}^{-1} \mathbf{c}_0

    Parameters
    ----------
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    cov_model : str
        ``"exponential"`` or ``"gaussian"``.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the kriging variance.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "Even the smallest person can change the course of the future."
        -- Galadriel, LOTR
    """
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _cov(h):
        if cov_model == "gaussian":
            return sill * np.exp(-((h / rng) ** 2)) + nug * (h == 0)
        return sill * np.exp(-h / rng) + nug * (h == 0)

    n = len(coords)
    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    C = _cov(dist_obs)
    d0 = np.sqrt(((coords - target) ** 2).sum(-1))
    c0 = _cov(d0)
    w = np.linalg.solve(C, c0)
    var = sill + nug - c0 @ w

    return SpatialResult(
        name="simple_kriging_variance",
        statistic=float(max(var, 0.0)),
        p_value=None,
        extra={"c0": c0, "weights": w},
    )


sgskv = simple_kriging_variance


def cheatsheet() -> str:
    return "simple_kriging_variance({}) -> Simple kriging variance."
