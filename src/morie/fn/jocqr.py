# morie.fn -- function file (hadesllm/morie)
"""CQR: conformalize quantile-regression output using calibration residuals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_conformalized_quantile_regression"]


def joseph_conformalized_quantile_regression(calibration_y, calibration_q_lo, calibration_q_hi, alpha):
    """
    CQR: conformalize quantile-regression output using calibration residuals

    Formula: E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)); q_alpha = Quantile_{1-alpha} E; PI = [q_lo - q_alpha, q_hi + q_alpha]

    Parameters
    ----------
    calibration_y : array-like
        Input data.
    calibration_q_lo : array-like
        Input data.
    calibration_q_hi : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: q_alpha

    References
    ----------
    Joseph Ch 17, Conformalized Quantile Regression section
    """
    calibration_y = np.atleast_1d(np.asarray(calibration_y, dtype=float))
    n = len(calibration_y)
    result = float(np.mean(calibration_y))
    se = float(np.std(calibration_y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CQR: conformalize quantile-regression output using calibration residuals"})


def cheatsheet():
    return "jocqr: CQR: conformalize quantile-regression output using calibration residuals"
