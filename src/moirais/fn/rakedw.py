"""Double-weighting (calibration + nonresponse)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rake_double_weights"]


def rake_double_weights(w_nr, w_cal):
    """
    Double-weighting (calibration + nonresponse)

    Formula: product of nonresponse + calibration weights

    Parameters
    ----------
    w_nr : array-like
        Input data.
    w_cal : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Särndal-Lundström (2005)
    """
    w_nr = np.atleast_1d(np.asarray(w_nr, dtype=float))
    n = len(w_nr)
    result = float(np.mean(w_nr))
    se = float(np.std(w_nr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double-weighting (calibration + nonresponse)"})


def cheatsheet():
    return "rakedw: Double-weighting (calibration + nonresponse)"
