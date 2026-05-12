# morie.fn -- function file (hadesllm/morie)
"""Beta kernel density estimator for bounded support."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kbeta"]


def kbeta(
    data: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bw: float | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
    n_grid: int = 512,
) -> dict:
    r"""
    Beta kernel density estimator for data on a bounded interval.

    Uses asymmetric beta kernels that naturally respect boundaries:

    .. math::

        \hat{f}(x) = \frac{1}{n} \sum_{i=1}^{n}
        \frac{t^{X_i/b}(1-t)^{(1-X_i)/b}}{B(X_i/b+1,\,(1-X_i)/b+1)}

    where :math:`t = (x - a)/(b_u - a)` is the rescaled evaluation
    point and *b* is the bandwidth.

    Parameters
    ----------
    data : np.ndarray
        1-d observations in [lower, upper].
    x_eval : np.ndarray or None
        Evaluation points within the support.
    bw : float or None
        Bandwidth parameter. Rule-of-thumb if None.
    lower : float
        Lower bound. Default 0.
    upper : float
        Upper bound. Default 1.
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``bw``.

    Raises
    ------
    ValueError
        If data outside [lower, upper].

    References
    ----------
    Chen, S. X. (1999). Beta kernel estimators for density functions.
        *Computational Statistics & Data Analysis*, 31(2), 131-145.
    """
    from scipy.special import betaln

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    if lower >= upper:
        raise ValueError("lower must be < upper.")

    data_std = (data - lower) / (upper - lower)
    if np.any(data_std < 0) or np.any(data_std > 1):
        raise ValueError("All data must be within [lower, upper].")

    if bw is None:
        bw = max(np.std(data_std, ddof=1) * n ** (-0.4), 1e-6)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    if x_eval is None:
        x_eval = np.linspace(lower, upper, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    t = np.clip((x_eval - lower) / (upper - lower), 1e-10, 1 - 1e-10)

    density = np.zeros(len(x_eval))
    for i in range(n):
        xi = np.clip(data_std[i], 1e-10, 1 - 1e-10)
        a_param = xi / bw + 1.0
        b_param = (1.0 - xi) / bw + 1.0
        log_k = ((a_param - 1) * np.log(t)
                 + (b_param - 1) * np.log(1 - t)
                 - betaln(a_param, b_param))
        density += np.exp(log_k)
    density /= n * (upper - lower)

    return RichResult(payload={"x_eval": x_eval, "density": density, "bw": bw})


def cheatsheet() -> str:
    return "kbeta({data}) -> Beta kernel density for bounded data."
