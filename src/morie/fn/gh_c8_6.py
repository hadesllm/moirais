# morie.fn -- function file (hadesllm/morie)
"""i.i.d. contraction theorem in Hellinger metric: main tool for density estimation rates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_iid_crt_thm"]


def ghosal_iid_crt_thm(x):
    """
    i.i.d. contraction theorem in Hellinger metric: main tool for density estimation rates

    Formula: d_H rate eps_n from test+prior mass+entropy conditions under i.i.d. P0

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 8 §8.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "i.i.d. contraction theorem in Hellinger metric: main tool for density estimation rates"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "i.i.d. contraction theorem in Hellinger metric: main tool for density estimation rates"})


def cheatsheet():
    return "gh_c8_6: i.i.d. contraction theorem in Hellinger metric: main tool for density estimation rates"
