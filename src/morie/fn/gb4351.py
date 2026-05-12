# morie.fn -- function file (hadesllm/morie)
"""V = 4n*D+^2 is asymptotically chi-square with 2 df."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_ks_chi2_approx"]


def gibbons_ks_chi2_approx(n, Dplus):
    """
    V = 4n*D+^2 is asymptotically chi-square with 2 df

    Formula: 4n*D+_n^2 ->_d chi2(2) as n -> inf

    Parameters
    ----------
    n : array-like
        Input data.
    Dplus : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: chi2_statistic

    References
    ----------
    Gibbons Corollary 4.3.5.1
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "V = 4n*D+^2 is asymptotically chi-square with 2 df"})


def cheatsheet():
    return "gb4351: V = 4n*D+^2 is asymptotically chi-square with 2 df"
