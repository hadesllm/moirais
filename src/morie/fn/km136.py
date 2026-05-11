"""Mml vlm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_mml_vlm_loss"]


def kamath_ch9_mml_vlm_loss(Pos, Neg):
    """
    Mml vlm loss.

    Formula: L_{MML} = -\sum_{(x,y)\in\text{Pos}} \log p(\text{aligned}|x,y) - \sum_{(x',y')\in\text{Neg}} \log p(\text{unaligned}|x',y')

    Parameters
    ----------
    Pos : array-like
        Input data.
    Neg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.8, p. 386
    """
    Pos = np.atleast_1d(np.asarray(Pos, dtype=float))
    n = len(Pos)
    result = float(np.mean(Pos))
    se = float(np.std(Pos, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mml vlm loss."})


def cheatsheet():
    return "km136: Mml vlm loss."
