# morie.fn — function file (hadesllm/morie)
"""Glivenko-Cantelli theorem verification (Kosorok 2008, Ch 2).

Computes sup_t |F_n(t) - F(t)| against a hypothesised CDF F (default
standard normal), i.e. the one-sample Kolmogorov-Smirnov statistic,
plus its exact KS asymptotic p-value (Marsaglia-Tsang-Wang series).
By Glivenko-Cantelli the statistic -> 0 a.s. when F is correct.
"""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_glivenko_cantelli"]


def kosorok_glivenko_cantelli(x, cdf="norm"):
    """One-sample KS-style sup|F_n - F| statistic.

    Parameters
    ----------
    x : array-like
        IID sample.
    cdf : str or callable
        Name of a `scipy.stats` distribution (default 'norm', standard
        normal) or any callable F : R -> [0,1].

    Returns
    -------
    RichResult with keys: statistic, p_value, n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    res = stats.kstest(x, cdf)
    return RichResult(payload={
        "statistic": float(res.statistic),
        "p_value":   float(res.pvalue),
        "n":         n,
        "method":    "Glivenko-Cantelli / KS sup|F_n - F|",
    })


def cheatsheet():
    return "ksr03: Glivenko-Cantelli sup|F_n - F| (one-sample KS)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(kosorok_glivenko_cantelli(rng.normal(size=200)))
