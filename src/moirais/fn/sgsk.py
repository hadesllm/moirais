"""Simple kriging interpolation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def simple_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    mean: float = 0.0,
) -> SpatialResult:
    r"""Simple kriging prediction assuming known constant mean.

    .. math::

        \hat{Z}(s_0) = \mu + \mathbf{c}_0^T \mathbf{C}^{-1} (\mathbf{Z} - \mu)

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)`` or ``(m, 2)``.
    cov_model : str
        Covariance model: ``"exponential"`` or ``"gaussian"``.
    cov_params : dict, optional
        ``{"sill": float, "range": float, "nugget": float}``.
    mean : float
        Known process mean.

    Returns
    -------
    SpatialResult
        ``statistic`` is the predicted value (first target).
        ``extra`` contains ``predictions``, ``variances``, ``weights``.

    References
    ----------
    Schabenberger O, Gotway CA (2005). *Statistical Methods for Spatial
    Data Analysis*. Chapman & Hall/CRC, Ch. 5.

    .. epigraph::

        "One does not simply walk into Mordor." -- Boromir, LOTR
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    n = len(Z)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _cov(h):
        if cov_model == "gaussian":
            return sill * np.exp(-((h / rng) ** 2)) + nug * (h == 0)
        return sill * np.exp(-h / rng) + nug * (h == 0)

    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    C = _cov(dist_obs)

    preds, varis, all_w = [], [], []
    C_inv = np.linalg.solve(C, np.eye(n))
    residuals = Z - mean

    for t in target:
        d0 = np.sqrt(((coords - t) ** 2).sum(-1))
        c0 = _cov(d0)
        w = C_inv @ c0
        pred = mean + w @ residuals
        var = sill + nug - c0 @ w
        preds.append(float(pred))
        varis.append(float(max(var, 0.0)))
        all_w.append(w)

    return SpatialResult(
        name="simple_kriging",
        statistic=preds[0],
        p_value=None,
        extra={
            "predictions": np.array(preds),
            "variances": np.array(varis),
            "weights": all_w,
        },
    )


sgsk = simple_kriging


def cheatsheet() -> str:
    return "simple_kriging({}) -> Simple kriging interpolation."
