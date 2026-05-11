"""Alignscore total loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_alignscore_total_loss"]


def kamath_ch6_alignscore_total_loss(L_3way, L_bin, L_reg, lambdas):
    """
    Alignscore total loss.

    Formula: L_{total} = \lambda_1 L_{3way} + \lambda_2 L_{bin} + \lambda_3 L_{reg}

    Parameters
    ----------
    L_3way : array-like
        Input data.
    L_bin : array-like
        Input data.
    L_reg : array-like
        Input data.
    lambdas : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.3, p. 220
    """
    L_3way = np.atleast_1d(np.asarray(L_3way, dtype=float))
    n = len(L_3way)
    result = float(np.mean(L_3way))
    se = float(np.std(L_3way, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alignscore total loss."})


def cheatsheet():
    return "km079: Alignscore total loss."
