"""Disjunctive kriging via Hermite polynomials."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def disjunctive_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    n_hermite: int = 10,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Disjunctive kriging using Hermite polynomial expansion.

    Expands the prediction as a sum of Hermite polynomial terms
    of the normal-score transformed data.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    n_hermite : int
        Number of Hermite polynomial terms.
    vario_model : str
        Variogram model.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the disjunctive kriging prediction.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "I will take the Ring to Mordor." -- Frodo, LOTR
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(Z)

    ranks = np.argsort(np.argsort(Z))
    from scipy.stats import norm

    Y = norm.ppf((ranks + 0.5) / n)

    def _hermite(y, k):
        if k == 0:
            return np.ones_like(y)
        if k == 1:
            return y
        h_prev, h_curr = np.ones_like(y), y.copy()
        for j in range(2, k + 1):
            h_next = y * h_curr - (j - 1) * h_prev
            h_prev, h_curr = h_curr, h_next
        return h_curr

    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = _gamma(dist_obs)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    d0 = np.sqrt(((coords - target) ** 2).sum(-1))
    g0 = _gamma(d0)
    b = np.zeros(n + 1)
    b[:n] = g0
    b[n] = 1.0
    lam = np.linalg.solve(A, b)
    w = lam[:n]

    pred = 0.0
    for k in range(n_hermite):
        hk = _hermite(Y, k)
        import math

        coeff = np.mean(Z * hk) / max(math.factorial(k), 1)
        y0_hat = w @ Y
        pred += coeff * _hermite(np.array([y0_hat]), k)[0]

    return SpatialResult(
        name="disjunctive_kriging",
        statistic=float(pred),
        p_value=None,
        extra={"n_hermite": n_hermite, "normal_score_pred": float(w @ Y)},
    )


sgdjk = disjunctive_kriging


def cheatsheet() -> str:
    return "disjunctive_kriging({}) -> Disjunctive kriging via Hermite polynomials."
