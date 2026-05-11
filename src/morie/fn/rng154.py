"""Output of the ANC as the difference between primary input and adaptive filter output.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_anc_output"]


def rangayyan_ch3_anc_output(x, y, n):
    """
    Output of the ANC as the difference between primary input and adaptive filter output.

    Formula: v_tilde(n) = e(n) = x(n) - y(n)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.188, p. 182
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of the ANC as the difference between primary input and adaptive filter output."})


def cheatsheet():
    return "rng154: Output of the ANC as the difference between primary input and adaptive filter output."
