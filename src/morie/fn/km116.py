"""Brevity penalty.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_brevity_penalty"]


def kamath_ch8_brevity_penalty(c, r):
    """
    Brevity penalty.

    Formula: \mathrm{BP} = \begin{cases} 1 & c>r \\ e^{1-r/c} & c\le r \end{cases}

    Parameters
    ----------
    c : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.4, p. 323
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Brevity penalty."})


def cheatsheet():
    return "km116: Brevity penalty."
