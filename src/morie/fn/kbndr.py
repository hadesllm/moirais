# morie.fn -- function file (hadesllm/morie)
"""Boundary-corrected kernel density estimator."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kbndr"]


def kbndr(data: np.ndarray, x_eval: np.ndarray | None = None, cdf=None, *, bw: float | None = None, lower: float = 0.0, upper: float = np.inf, method: str = "reflection", n_grid: int = 512) -> dict:
    r"""
    Boundary-corrected kernel density estimator.

    Applies the reflection method to eliminate boundary bias:

    .. math::

        \hat{f}_{\mathrm{refl}}(x) = \hat{f}(x) + \hat{f}(2a - x)

    where *a* is the lower boundary. For two-sided boundaries, reflections
    from both *a* and *b* are used with renormalization.

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations within [lower, upper].
    x_eval : np.ndarray or None
        Grid of evaluation points within the support.
    bw : float or None
        Bandwidth. Silverman's rule if None.
    lower : float
        Lower boundary. Default 0.
    upper : float
        Upper boundary. Default inf (one-sided correction).
    method : str
        ``'reflection'`` (default) or ``'renormalization'``.
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``bw``, ``method``.

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
        Analysis*. Chapman & Hall. Section 2.10.
    """
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
        lo = max(lower, data.min())
        hi = data.max() if np.isinf(upper) else min(upper, data.max())
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    def _gauss_kde(pts, d):
        u = (pts[:, None] - d[None, :]) / bw
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return k.mean(axis=1) / bw

    if method == "reflection":
        augmented = [data]
        if not np.isneginf(lower):
            augmented.append(2 * lower - data)
        if not np.isinf(upper):
            augmented.append(2 * upper - data)
        aug = np.concatenate(augmented)
        density = _gauss_kde(x_eval, aug) * (len(augmented))
        mask = np.ones_like(x_eval, dtype=bool)
        if not np.isneginf(lower):
            mask &= x_eval >= lower
        if not np.isinf(upper):
            mask &= x_eval <= upper
        density[~mask] = 0.0
    elif method == "renormalization":
        from scipy.stats import norm
        density = _gauss_kde(x_eval, data)
        a_lo = (lower - data[None, :]) / bw if not np.isneginf(lower) else -np.inf
        a_hi = (upper - data[None, :]) / bw if not np.isinf(upper) else np.inf
        if np.isscalar(a_lo):
            a_lo = np.full((1, n), a_lo)
        if np.isscalar(a_hi):
            a_hi = np.full((1, n), a_hi)
        norm_factor = (norm.cdf(a_hi) - norm.cdf(a_lo)).mean(axis=1)
        norm_factor = np.maximum(norm_factor, 1e-15)
        density /= norm_factor
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'reflection' or 'renormalization'.")

    return RichResult(payload={"x_eval": x_eval, "density": density, "bw": bw, "method": method})


def cheatsheet() -> str:
    return "kbndr({data}) -> Boundary-corrected KDE (reflection/renormalization)."
