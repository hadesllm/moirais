"""Universal kriging interpolation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def universal_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
    trend_order: int = 1,
) -> SpatialResult:
    r"""Universal kriging with polynomial drift.

    Extends ordinary kriging by modelling the mean as a polynomial
    in the spatial coordinates.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)`` or ``(m, 2)``.
    vario_model : str
        ``"exponential"`` or ``"gaussian"``.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    trend_order : int
        Polynomial order (0=OK, 1=linear, 2=quadratic).

    Returns
    -------
    SpatialResult
        ``statistic`` is the prediction; ``extra`` has ``predictions``,
        ``variances``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    n = len(Z)
    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        if vario_model == "gaussian":
            return nug + sill * (1.0 - np.exp(-((h / rng) ** 2))) * (h > 0)
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    def _trend(c):
        cols = [np.ones(len(c))]
        if trend_order >= 1:
            cols.extend([c[:, 0], c[:, 1]])
        if trend_order >= 2:
            cols.extend([c[:, 0] ** 2, c[:, 0] * c[:, 1], c[:, 1] ** 2])
        return np.column_stack(cols)

    F = _trend(coords)
    p = F.shape[1]

    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = _gamma(dist_obs)

    K = n + p
    A = np.zeros((K, K))
    A[:n, :n] = G
    A[:n, n:] = F
    A[n:, :n] = F.T

    preds, varis = [], []
    for t in target:
        d0 = np.sqrt(((coords - t) ** 2).sum(-1))
        g0 = _gamma(d0)
        f0 = _trend(t.reshape(1, -1)).ravel()
        b = np.zeros(K)
        b[:n] = g0
        b[n:] = f0
        lam = np.linalg.solve(A, b)
        w = lam[:n]
        pred = w @ Z
        var = w @ g0 + lam[n:] @ f0
        preds.append(float(pred))
        varis.append(float(max(var, 0.0)))

    return SpatialResult(
        name="universal_kriging",
        statistic=preds[0],
        p_value=None,
        extra={
            "predictions": np.array(preds),
            "variances": np.array(varis),
            "trend_order": trend_order,
        },
    )


sguk = universal_kriging


def cheatsheet() -> str:
    return "universal_kriging({}) -> Universal kriging interpolation."
