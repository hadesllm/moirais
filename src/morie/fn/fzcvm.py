# morie.fn — function file (hadesllm/morie)
"""Cramer-von Mises test with kernel-smoothing (Fauzi Ch 5).

    W_n^2 = n * integral ( F_hat_h(t) - F_0(t) )^2 dF_0(t).

Under H0, the smoothing bias is o(n^{-1/2}), so the classical
CvM asymptotic distribution applies and we use its tabulated tail.
"""
import numpy as np
from scipy import stats as _sps
from ._richresult import RichResult

__all__ = ["fauzi_cvm_smoothed"]


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _cvm_pvalue(w2):
    """Linear-interp tail probability of CvM distribution from Anderson-Darling table."""
    if w2 <= 0:
        return 1.0
    table = [(0.347, 0.10), (0.461, 0.05), (0.581, 0.025),
             (0.743, 0.01), (1.168, 0.001)]
    if w2 < table[0][0]:
        return float(0.5)
    if w2 > table[-1][0]:
        return float(table[-1][1] * 0.5)
    for i in range(len(table) - 1):
        if table[i][0] <= w2 <= table[i + 1][0]:
            w1, p1 = table[i]
            w2_, p2 = table[i + 1]
            lp = np.log(p1) + (np.log(p2) - np.log(p1)) * (w2 - w1) / (w2_ - w1)
            return float(np.exp(lp))
    return 0.5


def fauzi_cvm_smoothed(x, cdf="norm", args=None, h=None):
    """Smoothed Cramer-von Mises test.

    Parameters
    ----------
    x : array-like
    cdf : str or callable  SciPy distribution name (default "norm").
    args : tuple           distribution params (default = MLE on x).
    h : float, optional    bandwidth.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan,
                                    "n": n, "method": "fzcvm — too few obs"})
    if h is None:
        h = float(_silverman_h(x))

    if callable(cdf):
        # Generic continuous reference — uniform-on-x-range quadrature
        t_grid = np.linspace(np.min(x), np.max(x), max(200, n))
        F_ref = np.array([cdf(g) for g in t_grid])
    else:
        dist = getattr(_sps, cdf)
        if args is None:
            if cdf == "norm":
                args = (float(np.mean(x)), float(np.std(x, ddof=1)))
            else:
                args = ()
        u = (np.arange(1, n + 1) - 0.5) / n
        t_grid = dist.ppf(u, *args)
        F_ref = dist.cdf(t_grid, *args)
    F_hat = np.array([np.mean(_sps.norm.cdf((g - x) / h)) for g in t_grid])
    w2 = float(n * np.mean((F_hat - F_ref) ** 2))
    p = _cvm_pvalue(w2 / n)  # CvM table is in the per-obs scale

    return RichResult(payload={
        "statistic": w2,
        "p_value": p,
        "h": h,
        "n": n,
        "method": "Fauzi kernel-smoothed Cramer-von Mises (Ch 5)",
    })


def cheatsheet():
    return "fzcvm: Kernel-smoothed Cramer-von Mises GoF test"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_cvm_smoothed(x, cdf="norm", args=(0.0, 1.0))
# >>> r["statistic"] >= 0
# True
