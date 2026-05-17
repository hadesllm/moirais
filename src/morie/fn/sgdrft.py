"""Kriging with external drift (KED)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def external_drift_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    drift_var: np.ndarray,
    drift_target: float | None = None,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Kriging with external drift variable.

    Uses an auxiliary variable as the drift function instead of
    polynomial trend.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    drift_var : np.ndarray
        External drift values at observation locations, shape ``(n,)``.
    drift_target : float, optional
        Drift value at target. Defaults to mean of ``drift_var``.
    vario_model : str
        Variogram model.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the KED prediction.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    dv = np.asarray(drift_var, dtype=np.float64).ravel()
    n = len(Z)
    if drift_target is None:
        drift_target = float(np.mean(dv))

    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = _gamma(dist)

    K = n + 2
    A = np.zeros((K, K))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[:n, n + 1] = dv
    A[n + 1, :n] = dv

    d0 = np.sqrt(((coords - target) ** 2).sum(-1))
    g0 = _gamma(d0)
    b = np.zeros(K)
    b[:n] = g0
    b[n] = 1.0
    b[n + 1] = drift_target

    lam = np.linalg.solve(A, b)
    pred = float(lam[:n] @ Z)
    var = float(lam[:n] @ g0 + lam[n] + lam[n + 1] * drift_target)

    return SpatialResult(
        name="external_drift_kriging",
        statistic=pred,
        p_value=None,
        extra={"variance": max(var, 0.0), "drift_target": drift_target},
    )


sgdrft = external_drift_kriging


def cheatsheet() -> str:
    return "external_drift_kriging({}) -> Kriging with external drift (KED)."
