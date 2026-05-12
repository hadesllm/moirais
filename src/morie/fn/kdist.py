# morie.fn -- function file (hadesllm/morie)
"""Kernel distribution function estimator."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kdist"]


def _silverman_bw(x: np.ndarray, cdf=None) -> float:
    n = x.shape[0]
    sigma = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    return max(0.9 * s * n ** (-0.2), 1e-10)


def kdist(
    data: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bw: float | None = None,
    n_grid: int = 512,
) -> dict:
    r"""
    Kernel distribution function estimator.

    Estimates the cumulative distribution function via:

    .. math::

        \hat{F}(x) = \frac{1}{n} \sum_{i=1}^{n}
        \Phi\!\left(\frac{x - X_i}{h}\right)

    where :math:`\Phi` is the standard normal CDF and *h* is the bandwidth.

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    x_eval : np.ndarray or None
        Grid of evaluation points. If None, *n_grid* equally spaced points are used.
    bw : float or None
        Bandwidth. If None, Silverman's rule of thumb is applied.
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``cdf``, ``bw``, ``n``.

    Raises
    ------
    ValueError
        If fewer than 2 observations or non-positive bandwidth.

    References
    ----------
    Azzalini, A. (1981). A note on the estimation of a distribution function
        and quantiles by a kernel method. *Biometrika*, 68(1), 326-328.
    """
    from scipy.stats import norm

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    if bw is None:
        bw = _silverman_bw(data)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    if x_eval is None:
        lo = data.min() - 3 * bw
        hi = data.max() + 3 * bw
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    u = (x_eval[:, None] - data[None, :]) / bw
    cdf = norm.cdf(u).mean(axis=1)

    return RichResult(payload={"x_eval": x_eval, "cdf": cdf, "bw": bw, "n": n})


def cheatsheet() -> str:
    return "kdist({data}) -> Kernel CDF estimator (Gaussian kernel)."
