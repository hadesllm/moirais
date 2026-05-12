# morie.fn — function file (hadesllm/morie)
"""Bias-reduced KDFE via geometric extrapolation (Fauzi Ch 2).

Geometric/Richardson-style extrapolation cancels the O(h^2) bias term
of the KDFE.  Given two bandwidths h and c*h (c>1):

    F_hat_br(t) = ( c^2 * F_hat_h(t) - F_hat_{c*h}(t) ) / ( c^2 - 1 )

This kills the leading h^2 bias term, leaving an O(h^4) residual at
the cost of a slight variance inflation by factor (c^4 + 1)/(c^2 - 1)^2.
"""
import numpy as np
from scipy import stats as _sps
from ._richresult import RichResult

__all__ = ["fauzi_bias_reduced_kdfe"]


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


def fauzi_bias_reduced_kdfe(x, t=None, h=None, c=2.0):
    """Bias-reduced KDFE at ``t`` via geometric extrapolation.

    Parameters
    ----------
    x : array-like
    t : float, optional   default = sample median
    h : float, optional   default = Silverman's rule
    c : float, optional   extrapolation ratio (>1). default 2.

    Returns
    -------
    RichResult with estimate (F_br), F_h, F_ch, h, c, n.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "BR-KDFE — too few obs"})
    if t is None:
        t = float(np.median(x))
    if h is None:
        h = float(_silverman_h(x))
    if c <= 1.0:
        raise ValueError("c must be > 1 for bias-reduction extrapolation")

    F_h = _kdfe(x, t, h)
    F_ch = _kdfe(x, t, c * h)
    F_br = (c * c * F_h - F_ch) / (c * c - 1.0)

    # Variance inflation factor for the BR estimator (approx)
    var_F = F_h * (1.0 - F_h) / n
    var_inflate = (c ** 4 + 1.0) / (c * c - 1.0) ** 2
    se = np.sqrt(var_F * var_inflate)

    return RichResult(payload={
        "estimate": F_br,
        "F_h": F_h,
        "F_ch": F_ch,
        "se": se,
        "h": h,
        "c": c,
        "t": t,
        "n": n,
        "method": "Fauzi bias-reduced KDFE (Ch 2)",
    })


def cheatsheet():
    return "fzbrd: Bias-reduced KDFE via Richardson-style extrapolation"


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_bias_reduced_kdfe(x, t=0.0)
# >>> abs(r["estimate"] - 0.5) < 0.1
# True
