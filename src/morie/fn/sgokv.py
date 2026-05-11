"""Ordinary kriging variance."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def ordinary_kriging_variance(
    coords: np.ndarray,
    target: np.ndarray,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Compute ordinary kriging variance.

    .. math::

        \sigma_{OK}^2(s_0) = \sum_i \lambda_i \gamma(s_i - s_0) + \mu

    Parameters
    ----------
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    vario_model : str
        Variogram model name.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the kriging variance.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "I'm going on an adventure!" -- Bilbo, LOTR
    """
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(coords)
    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        if vario_model == "gaussian":
            return nug + sill * (1.0 - np.exp(-((h / rng) ** 2))) * (h > 0)
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
    var = lam[:n] @ g0 + lam[n]

    return SpatialResult(
        name="ordinary_kriging_variance",
        statistic=float(max(var, 0.0)),
        p_value=None,
        extra={"lagrange_multiplier": float(lam[n])},
    )


sgokv = ordinary_kriging_variance


def cheatsheet() -> str:
    return "ordinary_kriging_variance({}) -> Ordinary kriging variance."
