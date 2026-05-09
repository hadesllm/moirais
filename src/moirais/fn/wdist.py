"""Wasserstein distance (1-D optimal transport)."""

__all__ = ["wdist"]

import numpy as np
from ._richresult import RichResult


def wdist(
    x: np.ndarray,
    y: np.ndarray,
    *,
    p: int = 1,
) -> dict:
    """
    Compute the p-Wasserstein distance between two 1-D distributions.

    For empirical distributions on the real line:

    .. math::

        W_p(\\mu, \\nu) = \\left(\\int_0^1 |F^{-1}(t) - G^{-1}(t)|^p
        \\, dt \\right)^{1/p}

    which reduces to sorting and pairing for equal-length samples.

    Parameters
    ----------
    x : np.ndarray
        First sample, shape (n,).
    y : np.ndarray
        Second sample, shape (m,).
    p : int
        Order of the Wasserstein distance (1 or 2). Default 1.

    Returns
    -------
    dict
        'distance' (float), 'order' (int).

    Raises
    ------
    ValueError
        If inputs are empty or p < 1.

    References
    ----------
    Villani, C. (2009). Optimal Transport: Old and New. Springer.
    Ramdas, A. et al. (2017). On Wasserstein two-sample testing.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input arrays must be non-empty.")
    if p < 1:
        raise ValueError("p must be >= 1.")

    n, m = len(x), len(y)
    xs = np.sort(x)
    ys = np.sort(y)

    if n == m:
        dist = np.mean(np.abs(xs - ys) ** p) ** (1.0 / p)
    else:
        all_vals = np.sort(np.concatenate([xs, ys]))
        all_vals = np.unique(np.concatenate([
            all_vals,
            [all_vals[0] - 1],
            [all_vals[-1] + 1],
        ]))

        cdf_x = np.searchsorted(xs, all_vals, side="right") / n
        cdf_y = np.searchsorted(ys, all_vals, side="right") / m

        deltas = np.diff(all_vals)
        mid_diff = np.abs(cdf_x[:-1] - cdf_y[:-1]) ** p
        dist = float(np.sum(mid_diff * deltas) ** (1.0 / p))

    return RichResult(payload={"distance": float(dist), "order": p})
