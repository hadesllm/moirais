# morie.fn — function file (hadesllm/morie)
"""Smoothed sign test (Fauzi Ch 5).

Classical sign test counts S = #{X_i > theta_0}.  Fauzi's smoothed
version replaces the indicator with the integrated kernel:

    S_n = sum_i W( (X_i - theta_0) / h ),   W = integral K.

Under H0: median(X)=theta_0,  E[S_n]=n/2, Var[S_n]≈n/4, so
z = (S_n - n/2)/sqrt(n/4) ~ N(0,1).
"""
import numpy as np
from scipy import stats as _sps
from ._richresult import RichResult

__all__ = ["fauzi_smoothed_sign"]


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def fauzi_smoothed_sign(x, theta0=0.0, h=None, alternative="two-sided"):
    """Smoothed sign test of H0: median = theta0."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan,
                                    "n": n, "method": "fzsgn — too few obs"})
    if h is None:
        h = float(_silverman_h(x))

    S_n = float(np.sum(_sps.norm.cdf((x - theta0) / h)))
    z = (S_n - n / 2.0) / np.sqrt(n / 4.0)

    if alternative == "two-sided":
        p = 2.0 * (1.0 - _sps.norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1.0 - _sps.norm.cdf(z)
    elif alternative == "less":
        p = float(_sps.norm.cdf(z))
    else:
        raise ValueError("alternative must be two-sided/greater/less")

    return RichResult(payload={
        "statistic": S_n,
        "z": float(z),
        "p_value": float(p),
        "theta0": theta0,
        "h": h,
        "n": n,
        "method": f"Fauzi smoothed sign test ({alternative}) (Ch 5)",
    })


def cheatsheet():
    return "fzsgn: Smoothed sign test for the median"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_smoothed_sign(x, theta0=0.0)
# >>> r["p_value"] > 0.05
# True
