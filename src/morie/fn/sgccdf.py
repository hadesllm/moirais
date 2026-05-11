"""Conditional CDF via indicator kriging at multiple thresholds."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def conditional_cdf_indicator(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    thresholds: np.ndarray | None = None,
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Estimate the conditional CDF at a target location.

    Applies indicator kriging at each threshold to build
    :math:`\hat{F}(z | s_0)`.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)``.
    thresholds : np.ndarray, optional
        Cutoff values. Defaults to 10 quantiles.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is median of conditional CDF.
        ``extra`` has ``thresholds``, ``cdf_values``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "Master Chief? Mind telling me what you're doing?"
        -- Lord Hood, Halo
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(Z)

    if thresholds is None:
        thresholds = np.quantile(Z, np.linspace(0.1, 0.9, 9))
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)

    params = vario_params or {"sill": 0.25, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 0.25)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = _gamma(dist)
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

    cdf_vals = np.empty(len(thresholds))
    for i, c in enumerate(thresholds):
        ind = (c >= Z).astype(np.float64)
        cdf_vals[i] = np.clip(w @ ind, 0.0, 1.0)

    cdf_vals = np.maximum.accumulate(cdf_vals)

    return SpatialResult(
        name="conditional_cdf_indicator",
        statistic=float(np.median(cdf_vals)),
        p_value=None,
        extra={"thresholds": thresholds, "cdf_values": cdf_vals},
    )


sgccdf = conditional_cdf_indicator


def cheatsheet() -> str:
    return "conditional_cdf_indicator({}) -> Conditional CDF via indicator kriging at multiple thresholds"
