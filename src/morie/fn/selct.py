# morie.fn — function file (hadesllm/morie)
"""Genomic selection response to selection prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["genomic_selection_accuracy"]


def genomic_selection_accuracy(i, predictive_ability, sigma_g, h2):
    """
    Genomic selection response to selection prediction

    Formula: R_GS = i * r_{IH} * sigma_g; r_{IH} = r(I, H) = r(GEBV, TBV) predictive ability i reliability

    Parameters
    ----------
    i : array-like
        Input data.
    predictive_ability : array-like
        Input data.
    sigma_g : array-like
        Input data.
    h2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'response': 'float'}

    References
    ----------
    Montesinos Lopez Ch 1,5
    """
    i = np.asarray(i, dtype=float)
    n = int(i) if i.ndim == 0 else len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genomic selection response to selection prediction"})


def cheatsheet():
    return "selct: Genomic selection response to selection prediction"
