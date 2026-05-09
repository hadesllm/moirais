"""Itm hard negative.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_itm_hard_negative"]


def kamath_ch9_itm_hard_negative(Pos, HardNeg):
    """
    Itm hard negative.

    Formula: L_{ITM-hn} = -\sum_{(x,y)\in\text{Pos}}\log p(\text{aligned}|x,y) - \sum_{(x',y')\in\text{HardNeg}}\log p(\text{unaligned}|x',y')

    Parameters
    ----------
    Pos : array-like
        Input data.
    HardNeg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.9, p. 387
    """
    Pos = np.atleast_1d(np.asarray(Pos, dtype=float))
    n = len(Pos)
    result = float(np.mean(Pos))
    se = float(np.std(Pos, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Itm hard negative."})


def cheatsheet():
    return "km137: Itm hard negative."
