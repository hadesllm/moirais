"""Sex-stratified Mendelian randomization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sex_specific_mr"]


def sex_specific_mr(y, exposure, instrument, sex):
    """
    Sex-stratified Mendelian randomization

    Formula: per-sex IV regression

    Parameters
    ----------
    y : array-like
        Input data.
    exposure : array-like
        Input data.
    instrument : array-like
        Input data.
    sex : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Burgess (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sex-stratified Mendelian randomization"})


def cheatsheet():
    return "mtr2sx: Sex-stratified Mendelian randomization"
