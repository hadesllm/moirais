"""Moran's I test on regression residuals."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def moran_residual_test(residuals: np.ndarray, W: np.ndarray, cdf=None) -> SpatialResult:
    r"""Moran's I test for spatial autocorrelation in residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals, shape ``(n,)``.
    W : np.ndarray
        Spatial weights matrix, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is Moran's I; ``p_value`` from normal approx.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

        "I have nothing more to hide." -- Freya, God of War
    """
    from scipy.stats import norm

    e = np.asarray(residuals, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64)
    n = len(e)
    em = e - e.mean()
    ss = np.sum(em**2)
    if ss == 0:
        return SpatialResult(name="moran_residual_test", statistic=0.0, p_value=1.0)

    S0 = W.sum()
    I = float(n * (em @ W @ em) / (S0 * ss)) if S0 > 0 else 0.0
    EI = -1.0 / (n - 1)

    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)
    k = float(n * np.sum(em**4) / (ss**2))
    VI = (n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2) - k * ((n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2)) / (
        (n - 1) * (n - 2) * (n - 3) * S0**2
    ) - EI**2
    VI = max(VI, 1e-10)
    z = (I - EI) / np.sqrt(VI)
    p = float(2 * (1 - norm.cdf(abs(z))))

    return SpatialResult(
        name="moran_residual_test",
        statistic=I,
        p_value=p,
        extra={"expected": EI, "z_score": float(z)},
    )


sgrmr = moran_residual_test


def cheatsheet() -> str:
    return "moran_residual_test({}) -> Moran's I test on regression residuals."
