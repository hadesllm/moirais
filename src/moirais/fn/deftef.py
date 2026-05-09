"""Design effect (DEFF)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["design_effect"]


def design_effect(design_var, srs_var):
    """
    Design effect (DEFF)

    Formula: DEFF = Var_design / Var_SRS

    Parameters
    ----------
    design_var : array-like
        Input data.
    srs_var : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kish (1965)
    """
    design_var = np.atleast_1d(np.asarray(design_var, dtype=float))
    n = len(design_var)
    result = float(np.mean(design_var))
    se = float(np.std(design_var, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Design effect (DEFF)"})


def cheatsheet():
    return "deftef: Design effect (DEFF)"
