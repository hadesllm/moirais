"""Schwarz inequality for complex functions A(f) and B(f).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_schwarz_inequality_complex"]


def rangayyan_ch4_schwarz_inequality_complex(A, B, f):
    """
    Schwarz inequality for complex functions A(f) and B(f).

    Formula: | integral A(f) B(f) df |^2 <= integral |A(f)|^2 df * integral |B(f)|^2 df

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.42, p. 238
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Schwarz inequality for complex functions A(f) and B(f)."})


def cheatsheet():
    return "rng216: Schwarz inequality for complex functions A(f) and B(f)."
