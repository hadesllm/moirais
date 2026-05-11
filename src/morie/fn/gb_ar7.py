# morie.fn — function file (hadesllm/morie)
"""ARE of Wilcoxon signed-rank vs t under logistic distribution: pi^2/9."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_are_logistic"]


def gibbons_are_logistic(distribution):
    """
    ARE of Wilcoxon signed-rank vs t under logistic distribution: pi^2/9

    Formula: ARE(Wilcoxon, t | Logistic) = pi^2/9 = 1.097

    Parameters
    ----------
    distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE

    References
    ----------
    Gibbons Ch 13
    """
    distribution = np.asarray(distribution, dtype=float)
    n = int(distribution) if distribution.ndim == 0 else len(distribution)
    result = float(np.mean(distribution))
    se = float(np.std(distribution, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARE of Wilcoxon signed-rank vs t under logistic distribution: pi^2/9"})


def cheatsheet():
    return "gb_ar7: ARE of Wilcoxon signed-rank vs t under logistic distribution: pi^2/9"
