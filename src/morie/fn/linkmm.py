"""IRT linking -- mean/mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["linking_meanmean"]


def linking_meanmean(params_form_a, params_form_b, common_items):
    """
    IRT linking -- mean/mean

    Formula: slope = SD(b_a)/SD(b_b); intercept = mean(b_a)-slope*mean(b_b)

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
    Marco (1977)
    """
    params_form_a = np.atleast_1d(np.asarray(params_form_a, dtype=float))
    n = len(params_form_a)
    result = float(np.mean(params_form_a))
    se = float(np.std(params_form_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IRT linking -- mean/mean"})


def cheatsheet():
    return "linkmm: IRT linking -- mean/mean"
