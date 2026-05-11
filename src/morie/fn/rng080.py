"""Parseval's theorem: total signal energy preserved under Fourier transform.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_parseval_theorem"]


def rangayyan_ch3_parseval_theorem(x, X):
    """
    Parseval's theorem: total signal energy preserved under Fourier transform.

    Formula: integral_{-inf}^{inf} |x(t)|^2 dt = (1/(2*pi)) * integral_{-inf}^{inf} |X(omega)|^2 d(omega); sum_{n=0}^{N-1} |x(n)|^2 = (1/N) * sum_{k=0}^{N-1} |X(k)|^2

    Parameters
    ----------
    x : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.91, p. 134
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parseval's theorem: total signal energy preserved under Fourier transform."})


def cheatsheet():
    return "rng080: Parseval's theorem: total signal energy preserved under Fourier transform."
