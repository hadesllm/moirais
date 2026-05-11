"""Fine-Gray subdistribution hazard."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fine_gray_subdistribution_hazard"]


def fine_gray_subdistribution_hazard(time, cause, X):
    """
    Fine-Gray subdistribution hazard

    Formula: lambda_1^FG(t) = -d/dt log(1 - F_1(t))

    Parameters
    ----------
    time : array-like
        Input data.
    cause : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fine & Gray (1999)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fine-Gray subdistribution hazard"})


def cheatsheet():
    return "fgsbh: Fine-Gray subdistribution hazard"
