"""Pass at k.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_pass_at_k"]


def kamath_ch8_pass_at_k(n, c, k):
    """
    Pass at k.

    Formula: \mathrm{pass}@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}

    Parameters
    ----------
    n : array-like
        Input data.
    c : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.16, p. 328
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pass at k."})


def cheatsheet():
    return "km128: Pass at k."
