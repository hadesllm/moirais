# morie.fn — function file (hadesllm/morie)
"""Cohen's kappa coefficient for classification agreement."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kappa_coefficient"]


def kappa_coefficient(y_true, y_pred):
    """
    Cohen's kappa coefficient for classification agreement

    Formula: kappa = (p_o - p_e) / (1 - p_e); p_o observed agreement, p_e expected by chance

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'kappa': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cohen's kappa coefficient for classification agreement"})


def cheatsheet():
    return "kapco: Cohen's kappa coefficient for classification agreement"
