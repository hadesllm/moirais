r"""Rrf score.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch7_rrf_score"]


def kamath_ch7_rrf_score(r):
    r"""
    Rrf score.

    Formula: \mathrm{Score}_{RRF} = \frac{1}{r+60}

    Parameters
    ----------
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 7, Eq 7.1, p. 285
    r"""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rrf score."})


def cheatsheet():
    return "km110: Rrf score."
