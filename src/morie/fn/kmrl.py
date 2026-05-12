# morie.fn -- function file (hadesllm/morie)
"""Kernel mean residual life estimator."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kmrl"]


def kmrl(data: np.ndarray, x_eval: np.ndarray | None = None, cdf=None, *, bw: float | None = None, n_grid: int = 256) -> dict:
    r"""
    Kernel mean residual life (MRL) estimator.

    The MRL function is defined as:

    .. math::

        e(t) = E[X - t \mid X > t] = \frac{\int_t^\infty \bar{F}(u)\,du}{\bar{F}(t)}

    This implementation uses the kernel survival function estimator
    :math:`\hat{\bar{F}}(t) = 1 - \hat{F}(t)` based on Gaussian kernels.

    Parameters
    ----------
    data : np.ndarray
        1-d array of positive observations (e.g., lifetimes).
    x_eval : np.ndarray or None
        Points at which to evaluate the MRL. Defaults to a grid.
    bw : float or None
        Bandwidth. Silverman's rule if None.
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``mrl``, ``bw``, ``n``.

    Raises
    ------
    ValueError
        If data is too small.

    References
    ----------
    Chaubey, Y. P. & Sen, P. K. (1999). On smooth estimation of mean
        residual life. *Journal of Statistical Planning and Inference*,
        75(2), 223-236.
    """
    from scipy.stats import norm

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    if bw is None:
        bw = max(0.9 * s * n ** (-0.2), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    if x_eval is None:
        lo = max(data.min(), 0.0)
        hi = data.max()
        x_eval = np.linspace(lo, hi * 0.95, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    upper = data.max() + 5 * bw
    fine = np.linspace(0, upper, 2048)
    u_fine = (fine[:, None] - data[None, :]) / bw
    surv_fine = 1.0 - norm.cdf(u_fine).mean(axis=1)
    surv_fine = np.maximum(surv_fine, 1e-15)

    dt = fine[1] - fine[0]
    cum_surv = np.cumsum(surv_fine[::-1])[::-1] * dt

    surv_at_x = np.interp(x_eval, fine, surv_fine)
    cum_at_x = np.interp(x_eval, fine, cum_surv)
    mrl = cum_at_x / np.maximum(surv_at_x, 1e-15)

    return RichResult(payload={"x_eval": x_eval, "mrl": mrl, "bw": bw, "n": n})


def cheatsheet() -> str:
    return "kmrl({data}) -> Kernel mean residual life estimator."
