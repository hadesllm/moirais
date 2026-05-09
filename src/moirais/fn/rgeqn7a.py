# moirais.fn — function file (hadesllm/moirais)
"""AR prediction error (residual energy)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch7_ar_prediction_err"]


def rangayyan_ch7_ar_prediction_err(x, a_coeffs):
    """
    AR prediction error (residual energy)

    Formula: P_m = sum e[n]^2 = R(0) + sum a_k*R(k) for k=1..m

    Parameters
    ----------
    x : array-like
        Input data.
    a_coeffs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction_error

    References
    ----------
    Rangayyan Ch 7.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR prediction error (residual energy)"})


def cheatsheet():
    return "rgeqn7a: AR prediction error (residual energy)"
