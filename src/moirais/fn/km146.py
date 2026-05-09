"""Output projector mse.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_output_projector_mse"]


def kamath_ch9_output_projector_mse(H_X, tau_X, t):
    """
    Output projector mse.

    Formula: \arg\min_{OUT\_ALIGN_{T\to X}} L_{mse}(H_X, \tau_X(t))

    Parameters
    ----------
    H_X : array-like
        Input data.
    tau_X : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.18, p. 397
    """
    H_X = np.atleast_1d(np.asarray(H_X, dtype=float))
    n = len(H_X)
    result = float(np.mean(H_X))
    se = float(np.std(H_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output projector mse."})


def cheatsheet():
    return "km146: Output projector mse."
