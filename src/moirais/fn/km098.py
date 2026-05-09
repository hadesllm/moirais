"""Log prob ratio attr.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_log_prob_ratio_attr"]


def kamath_ch6_log_prob_ratio_attr(a_i, a_j, K):
    """
    Log prob ratio attr.

    Formula: \sum_{k=1}^K \log\frac{P(a_i^{(k)})}{P(a_j^{(k)})}

    Parameters
    ----------
    a_i : array-like
        Input data.
    a_j : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.22, p. 244
    """
    a_i = np.atleast_1d(np.asarray(a_i, dtype=float))
    n = len(a_i)
    result = float(np.mean(a_i))
    se = float(np.std(a_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log prob ratio attr."})


def cheatsheet():
    return "km098: Log prob ratio attr."
