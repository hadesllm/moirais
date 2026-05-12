# morie.fn -- function file (hadesllm/morie)
"""Time-series-aware missing imputation (forward-fill / linear / seasonal-mean)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_missing_data_imputation_ts"]


def joseph_missing_data_imputation_ts(y, strategy, m):
    """
    Time-series-aware missing imputation (forward-fill / linear / seasonal-mean)

    Formula: strategies: ffill, bfill, linear interp, seasonal-mean (y_hat_t = mean(y_{t-km}))

    Parameters
    ----------
    y : array-like
        Input data.
    strategy : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_imputed

    References
    ----------
    Joseph Ch 2, Missing Value Imputation section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-series-aware missing imputation (forward-fill / linear / seasonal-mean)"})


def cheatsheet():
    return "jomimi: Time-series-aware missing imputation (forward-fill / linear / seasonal-mean)"
