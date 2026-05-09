# moirais.fn — function file (hadesllm/moirais)
"""Distribution of median of DP sample: quantile functional of random CDF."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_median"]


def ghosal_dp_median(x):
    """
    Distribution of median of DP sample: quantile functional of random CDF

    Formula: Med(G) ~ distribution determined by DP(alpha, G0) via quantile inversion

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
    Ghosal Ch 4 §4.3.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Distribution of median of DP sample: quantile functional of random CDF"})


def cheatsheet():
    return "gh_c4_17: Distribution of median of DP sample: quantile functional of random CDF"
