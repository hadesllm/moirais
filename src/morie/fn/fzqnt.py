# morie.fn -- function file (hadesllm/morie)
"""Asymptotic distribution of kernel quantile estimator (Fauzi Ch 3).

Kernel quantile estimator: Q_hat(p) = F_hat_h^{-1}(p) where F_hat_h is
the KDFE.  Its asymptotic distribution is

    sqrt(n) * ( Q_hat(p) - Q(p) )  ->  N( 0, tau^2 / f(Q(p))^2 )

with tau^2 = p(1-p) (the binomial variance of the empirical CDF at Q).
"""
import numpy as np
from scipy import stats as _sps
from scipy.optimize import brentq as _brentq
from ._richresult import RichResult

__all__ = ["fauzi_kernel_quantile_asymptotic"]


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _kdfe(x, t, h):
    return float(np.mean(_sps.norm.cdf((t - x) / h)))


def _kde_density(x, t, h):
    return float(np.mean(_sps.norm.pdf((t - x) / h) / h))


def fauzi_kernel_quantile_asymptotic(x, p=0.5, h=None):
    """Kernel quantile estimator with asymptotic SE.

    Parameters
    ----------
    x : array-like
    p : float in (0,1)   probability level. default 0.5 (median).
    h : float, optional  bandwidth (Silverman by default).

    Returns
    -------
    RichResult with estimate, se, p, h, n.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                    "method": "fzqnt -- too few obs"})
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    if h is None:
        h = float(_silverman_h(x))

    lo, hi = float(np.min(x)), float(np.max(x))
    pad = 5.0 * h + 1e-9
    try:
        q_hat = _brentq(lambda t: _kdfe(x, t, h) - p, lo - pad, hi + pad)
    except Exception:
        q_hat = float(np.quantile(x, p))

    f_q = _kde_density(x, q_hat, h)
    if f_q <= 0:
        se = np.nan
    else:
        se = float(np.sqrt(p * (1.0 - p) / n) / f_q)

    return RichResult(payload={
        "estimate": float(q_hat),
        "se": se,
        "p": p,
        "h": h,
        "density_at_Q": f_q,
        "n": n,
        "method": "Fauzi kernel quantile (Ch 3) asymptotic N(0, p(1-p)/f(Q)^2)",
    })


def cheatsheet():
    return "fzqnt: Kernel quantile + asymptotic SE p(1-p)/(n f(Q)^2)"


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(1000)
# >>> r = fauzi_kernel_quantile_asymptotic(x, p=0.5)
# >>> abs(r["estimate"]) < 0.15  # median ≈ 0
# True
