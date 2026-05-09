"""Cokriging for multivariate spatial prediction."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def cokriging(
    Z_primary: np.ndarray,
    Z_secondary: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    vario_params_primary: dict | None = None,
    vario_params_cross: dict | None = None,
    vario_params_secondary: dict | None = None,
) -> SpatialResult:
    r"""Ordinary cokriging with two variables.

    Builds a joint kriging system using auto- and cross-variograms.

    Parameters
    ----------
    Z_primary : np.ndarray
        Primary variable values, shape ``(n,)``.
    Z_secondary : np.ndarray
        Secondary variable values, shape ``(n,)``.
    coords : np.ndarray
        Shared observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    vario_params_primary, vario_params_cross, vario_params_secondary : dict
        Each has ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the cokriging prediction for the primary variable.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "For Frodo." -- Aragorn, LOTR
    """
    Z1 = np.asarray(Z_primary, dtype=np.float64).ravel()
    Z2 = np.asarray(Z_secondary, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(Z1)

    p1 = vario_params_primary or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    pc = vario_params_cross or {"sill": 0.5, "range": 1.0, "nugget": 0.0}
    p2 = vario_params_secondary or {"sill": 1.0, "range": 1.0, "nugget": 0.0}

    def _gamma(h, p):
        return p["nugget"] + p["sill"] * (1.0 - np.exp(-h / p["range"])) * (h > 0)

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G11 = _gamma(dist, p1)
    G12 = _gamma(dist, pc)
    G22 = _gamma(dist, p2)

    K = 2 * n + 2
    A = np.zeros((K, K))
    A[:n, :n] = G11
    A[:n, n : 2 * n] = G12
    A[n : 2 * n, :n] = G12
    A[n : 2 * n, n : 2 * n] = G22
    A[:n, 2 * n] = 1.0
    A[2 * n, :n] = 1.0
    A[n : 2 * n, 2 * n + 1] = 1.0
    A[2 * n + 1, n : 2 * n] = 1.0

    d0 = np.sqrt(((coords - target) ** 2).sum(-1))
    g0_11 = _gamma(d0, p1)
    g0_12 = _gamma(d0, pc)

    b = np.zeros(K)
    b[:n] = g0_11
    b[n : 2 * n] = g0_12
    b[2 * n] = 1.0

    lam = np.linalg.solve(A, b)
    pred = lam[:n] @ Z1 + lam[n : 2 * n] @ Z2
    var = lam[:n] @ g0_11 + lam[n : 2 * n] @ g0_12 + lam[2 * n]

    return SpatialResult(
        name="cokriging",
        statistic=float(pred),
        p_value=None,
        extra={
            "variance": float(max(var, 0.0)),
            "weights_primary": lam[:n],
            "weights_secondary": lam[n : 2 * n],
        },
    )


sgcok = cokriging


def cheatsheet() -> str:
    return "cokriging({}) -> Cokriging for multivariate spatial prediction."
