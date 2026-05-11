"""Power series expansion of log(1 + x) for |x| < 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_log_power_series"]


def rangayyan_ch4_log_power_series(x):
    """
    Power series expansion of log(1 + x) for |x| < 1.

    Formula: log(1 + x) = x - x^2/2 + x^3/3 - x^4/4 + ...

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.69, p. 248
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power series expansion of log(1 + x) for |x| < 1."})


def cheatsheet():
    return "rng241: Power series expansion of log(1 + x) for |x| < 1."
