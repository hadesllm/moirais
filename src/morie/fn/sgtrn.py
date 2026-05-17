"""Trans-Gaussian kriging."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def trans_gaussian_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    transform: str = "log",
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Kriging on transformed data with back-transformation.

    Supports ``"log"`` and ``"sqrt"`` transforms.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    transform : str
        ``"log"`` or ``"sqrt"``.
    vario_model : str
        Variogram model for transformed data.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is back-transformed prediction.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(Z)

    if transform == "log":
        Y = np.log(np.maximum(Z, 1e-10))
    elif transform == "sqrt":
        Y = np.sqrt(np.maximum(Z, 0.0))
    else:
        raise ValueError(f"Unknown transform: {transform}")

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
    y_hat = w @ Y
    ok_var = w @ g0 + lam[n]

    if transform == "log":
        z_hat = float(np.exp(y_hat + 0.5 * ok_var))
    else:
        z_hat = float(y_hat**2 + max(ok_var, 0.0))

    return SpatialResult(
        name="trans_gaussian_kriging",
        statistic=z_hat,
        p_value=None,
        extra={
            "transform": transform,
            "transformed_prediction": float(y_hat),
            "ok_variance": float(max(ok_var, 0.0)),
        },
    )


sgtrn = trans_gaussian_kriging


def cheatsheet() -> str:
    return "trans_gaussian_kriging({}) -> Trans-Gaussian kriging."
