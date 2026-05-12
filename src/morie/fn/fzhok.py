# morie.fn -- function file (hadesllm/morie)
"""Higher-order kernel for bias reduction (Fauzi Ch 1).

A kernel K_r is of order r if

    integral K_r(u) du     = 1,
    integral u^j K_r(u) du = 0   for j = 1, …, r-1,
    integral u^r K_r(u) du != 0.

Higher-order kernels reduce the leading-order bias of a KDE from
O(h^2) (order-2) to O(h^r), at the cost of allowing K_r(u) < 0 (so the
density estimate is not guaranteed non-negative).

We implement the standard order-4 Gaussian-based kernel
(Wand & Jones 1995, eqn 2.8):

    K_4(u) = (1/2) * (3 - u^2) * phi(u)

where phi is the standard normal density.  Its second moment is 0
and fourth moment is -3.
"""
import numpy as np
from scipy import stats as _sps
from ._richresult import RichResult

__all__ = ["fauzi_higher_order_kernel"]


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _k4(u):
    """Order-4 Gaussian-based kernel (Wand-Jones eq 2.8)."""
    return 0.5 * (3.0 - u * u) * _sps.norm.pdf(u)


def fauzi_higher_order_kernel(x, t=None, h=None, order=4):
    """Kernel density estimate at ``t`` using an order-``order`` kernel.

    Currently only order=4 is implemented (the standard textbook case).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "fzhok -- too few obs"})
    if order != 4:
        raise NotImplementedError("only order=4 implemented (Wand-Jones eq 2.8)")
    if t is None:
        t = float(np.median(x))
    if h is None:
        h = float(_silverman_h(x))

    u = (t - x) / h
    f_hat = float(np.mean(_k4(u)) / h)

    # Order-4 kernel moments (analytic): mu_4 = -3 (Wand & Jones)
    mu4_K4 = -3.0
    R_K4 = 27.0 / (32.0 * np.sqrt(np.pi))  # integral K_4^2

    return RichResult(payload={
        "estimate": f_hat,
        "h": h,
        "t": t,
        "order": order,
        "mu_r": mu4_K4,
        "R_K": R_K4,
        "n": n,
        "method": "Fauzi higher-order (4) kernel density (Ch 1)",
    })


def cheatsheet():
    return "fzhok: Higher-order (4) kernel density estimate"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(2000)
# >>> r = fauzi_higher_order_kernel(x, t=0.0)
# >>> abs(r["estimate"] - 0.3989) < 0.1  # phi(0) = 1/sqrt(2pi) ≈ 0.3989
# True
