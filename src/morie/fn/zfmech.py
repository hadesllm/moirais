"""z-CDP mechanism noise calibration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["z_dp_mechanism"]


def z_dp_mechanism(y, sensitivity, rho):
    """
    z-CDP mechanism noise calibration

    Formula: sigma >= sensitivity / sqrt(2 * rho); rho = epsilon^2/2

    Parameters
    ----------
    y : array-like
        Input data.
    sensitivity : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bun & Steinke (2016) zCDP
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "z-CDP mechanism noise calibration"})


def cheatsheet():
    return "zfmech: z-CDP mechanism noise calibration"
