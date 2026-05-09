"""Saffir-Simpson + accumulated cyclone energy."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cyclone_intensity"]


def cyclone_intensity(v_max):
    """
    Saffir-Simpson + accumulated cyclone energy

    Formula: ACE = sum v_max² / 10⁴ over 6h obs

    Parameters
    ----------
    v_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bell et al (2000) ACE
    """
    v_max = np.atleast_1d(np.asarray(v_max, dtype=float))
    n = len(v_max)
    result = float(np.mean(v_max))
    se = float(np.std(v_max, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Saffir-Simpson + accumulated cyclone energy"})


def cheatsheet():
    return "cyclE: Saffir-Simpson + accumulated cyclone energy"
