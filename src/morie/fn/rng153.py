"""Primary input of an adaptive noise canceller (ANC): signal plus primary noise.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_anc_primary_input_model"]


def rangayyan_ch3_anc_primary_input_model(v, m, n):
    """
    Primary input of an adaptive noise canceller (ANC): signal plus primary noise.

    Formula: x(n) = v(n) + m(n)

    Parameters
    ----------
    v : array-like
        Input data.
    m : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.187, p. 181
    """
    v = np.atleast_1d(np.asarray(v, dtype=float))
    n = len(v)
    result = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Primary input of an adaptive noise canceller (ANC): signal plus primary noise."})


def cheatsheet():
    return "rng153: Primary input of an adaptive noise canceller (ANC): signal plus primary noise."
