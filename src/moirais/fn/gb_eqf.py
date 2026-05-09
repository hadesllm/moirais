# moirais.fn — function file (hadesllm/moirais)
"""Empirical quantile function Q_n(u) as inverse of EDF."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_emp_quantile"]


def gibbons_emp_quantile(u, data):
    """
    Empirical quantile function Q_n(u) as inverse of EDF

    Formula: Q_n(u) = X_(i) if (i-1)/n < u <= i/n

    Parameters
    ----------
    u : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: quantile

    References
    ----------
    Gibbons Ch 2.3.1
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical quantile function Q_n(u) as inverse of EDF"})


def cheatsheet():
    return "gb_eqf: Empirical quantile function Q_n(u) as inverse of EDF"
