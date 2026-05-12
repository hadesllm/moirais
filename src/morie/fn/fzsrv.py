# morie.fn -- function file (hadesllm/morie)
"""Kernel survival function estimator (Fauzi Ch 4).

For non-negative X:

    S_hat_h(t) = 1 - F_hat_h(t) = (1/n) * sum_i ( 1 - W((t-X_i)/h) ),

with W(u) = Phi(u) for the Gaussian kernel.  Asymptotic variance is
the KDFE variance: S(t)(1-S(t))/n + O(h/n).
"""
import numpy as np
from scipy import stats as _sps
from ._richresult import RichResult

__all__ = ["fauzi_survival_kernel"]


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def fauzi_survival_kernel(x, t=None, h=None):
    """Kernel survival estimate at ``t`` with asymptotic 95 percent CI.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "fzsrv -- too few obs"})
    if t is None:
        t = float(np.median(x))
    if h is None:
        h = float(_silverman_h(x))

    F_hat = float(np.mean(_sps.norm.cdf((t - x) / h)))
    S_hat = 1.0 - F_hat
    se = float(np.sqrt(S_hat * (1.0 - S_hat) / n))
    z = 1.959963984540054
    lo = max(0.0, S_hat - z * se)
    hi = min(1.0, S_hat + z * se)

    return RichResult(payload={
        "estimate": S_hat,
        "se": se,
        "ci_lower": lo,
        "ci_upper": hi,
        "t": t,
        "h": h,
        "n": n,
        "method": "Fauzi kernel survival S_hat(t)=1-F_hat_h(t) (Ch 4)",
    })


def cheatsheet():
    return "fzsrv: Kernel survival S_hat(t) = 1 - F_hat_h(t) + 95% CI"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.exponential(scale=1.0, size=2000)
# >>> r = fauzi_survival_kernel(x, t=1.0)
# >>> abs(r["estimate"] - np.exp(-1)) < 0.05  # S(1)=e^-1
# True
