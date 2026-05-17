"""Indicator kriging."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def indicator_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.0,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Indicator kriging for exceedance probability.

    Transforms observations to indicators :math:`I(Z(s) \le c)` and
    applies ordinary kriging to estimate :math:`P(Z(s_0) \le c)`.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    threshold : float
        Cutoff value *c*.
    vario_model : str
        Variogram model for indicator.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is :math:`\hat{P}(Z(s_0) \le c)`.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(Z)

    indicators = (threshold >= Z).astype(np.float64)
    params = vario_params or {"sill": 0.25, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 0.25)
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
    prob = float(np.clip(lam[:n] @ indicators, 0.0, 1.0))

    return SpatialResult(
        name="indicator_kriging",
        statistic=prob,
        p_value=None,
        extra={"threshold": threshold, "n_below": int(indicators.sum())},
    )


sgind = indicator_kriging


def cheatsheet() -> str:
    return "indicator_kriging({}) -> Indicator kriging."
