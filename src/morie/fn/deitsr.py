"""DeiT distillation token."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deit_distill"]


def deit_distill(x, teacher):
    """
    DeiT distillation token

    Formula: add second token; KD loss to teacher

    Parameters
    ----------
    x : array-like
        Input data.
    teacher : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Touvron et al (2021)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DeiT distillation token"})


def cheatsheet():
    return "deitsr: DeiT distillation token"
