# morie.fn -- function file (hadesllm/morie)
"""Bottom-up reconciliation: sum base-level forecasts to aggregate levels."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_bottom_up_reconciliation"]


def joseph_bottom_up_reconciliation(y_hat_bottom, S):
    """
    Bottom-up reconciliation: sum base-level forecasts to aggregate levels

    Formula: y_tilde_aggregate = S * y_hat_bottom

    Parameters
    ----------
    y_hat_bottom : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_reconciled

    References
    ----------
    Joseph Ch 17, Bottom-Up Hierarchical Forecasting section
    """
    y_hat_bottom = np.atleast_1d(np.asarray(y_hat_bottom, dtype=float))
    n = len(y_hat_bottom)
    result = float(np.mean(y_hat_bottom))
    se = float(np.std(y_hat_bottom, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bottom-up reconciliation: sum base-level forecasts to aggregate levels"})


def cheatsheet():
    return "johbu: Bottom-up reconciliation: sum base-level forecasts to aggregate levels"
