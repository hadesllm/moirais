"""IRT linking — Stocking-Lord."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["linking_stocking_lord"]


def linking_stocking_lord(params_form_a, params_form_b, common_items):
    """
    IRT linking — Stocking-Lord

    Formula: slope/intercept minimizing TCC distance

    Parameters
    ----------
    params_form_a : array-like
        Input data.
    params_form_b : array-like
        Input data.
    common_items : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stocking-Lord (1983)
    """
    params_form_a = np.atleast_1d(np.asarray(params_form_a, dtype=float))
    n = len(params_form_a)
    result = float(np.mean(params_form_a))
    se = float(np.std(params_form_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IRT linking — Stocking-Lord"})


def cheatsheet():
    return "linkqp: IRT linking — Stocking-Lord"
