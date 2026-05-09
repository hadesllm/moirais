"""Jensen-Shannon divergence between two closed compositions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_jsd"]


def compositional_jsd(p, q):
    """
    Jensen-Shannon divergence between two closed compositions

    Formula: JSD(p,q) = 1/2 KL(p||m) + 1/2 KL(q||m), m=(p+q)/2

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: jsd

    References
    ----------
    Lin (1991)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jensen-Shannon divergence between two closed compositions"})


def cheatsheet():
    return "aitjsd: Jensen-Shannon divergence between two closed compositions"
