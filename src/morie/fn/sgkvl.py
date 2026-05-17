"""Kriging cross-validation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def kriging_cross_validation(
    Z: np.ndarray,
    coords: np.ndarray,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
    method: str = "loo",
) -> SpatialResult:
    r"""Leave-one-out cross-validation for kriging.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    vario_model : str
        Variogram model.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    method : str
        ``"loo"`` for leave-one-out.

    Returns
    -------
    SpatialResult
        ``statistic`` is RMSE; ``extra`` has ``mae``, ``errors``,
        ``predictions``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(Z)
    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    errors = np.empty(n)
    preds = np.empty(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        c_sub = coords[mask]
        z_sub = Z[mask]
        ns = n - 1

        dist_sub = np.sqrt(((c_sub[:, None, :] - c_sub[None, :, :]) ** 2).sum(-1))
        G = _gamma(dist_sub)
        A = np.zeros((ns + 1, ns + 1))
        A[:ns, :ns] = G
        A[:ns, ns] = 1.0
        A[ns, :ns] = 1.0

        d0 = np.sqrt(((c_sub - coords[i]) ** 2).sum(-1))
        g0 = _gamma(d0)
        b = np.zeros(ns + 1)
        b[:ns] = g0
        b[ns] = 1.0
        lam = np.linalg.solve(A, b)
        preds[i] = lam[:ns] @ z_sub
        errors[i] = Z[i] - preds[i]

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    return SpatialResult(
        name="kriging_cross_validation",
        statistic=rmse,
        p_value=None,
        extra={"mae": mae, "errors": errors, "predictions": preds},
    )


sgkvl = kriging_cross_validation


def cheatsheet() -> str:
    return "kriging_cross_validation({}) -> Kriging cross-validation."
