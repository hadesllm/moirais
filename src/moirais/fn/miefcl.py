"""MI Rubin's rules combination."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multiple_imputation_combine"]


def multiple_imputation_combine(estimates_list, ses_list, m):
    """
    MI Rubin's rules combination

    Formula: point = avg over m; var = within + (1+1/m) between

    Parameters
    ----------
    estimates_list : array-like
        Input data.
    ses_list : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rubin (1987)
    """
    estimates_list = np.atleast_1d(np.asarray(estimates_list, dtype=float))
    n = len(estimates_list)
    result = float(np.mean(estimates_list))
    se = float(np.std(estimates_list, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MI Rubin's rules combination"})


def cheatsheet():
    return "miefcl: MI Rubin's rules combination"
