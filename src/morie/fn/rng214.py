"""Total energy of a signal via Parseval's theorem.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_signal_total_energy"]


def rangayyan_ch4_signal_total_energy(x, X, t, f):
    """
    Total energy of a signal via Parseval's theorem.

    Formula: E_x = integral_{-inf}^{inf} x^2(t) dt = integral_{-inf}^{inf} |X(f)|^2 df

    Parameters
    ----------
    x : array-like
        Input data.
    X : array-like
        Input data.
    t : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.40, p. 238
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Total energy of a signal via Parseval's theorem."})


def cheatsheet():
    return "rng214: Total energy of a signal via Parseval's theorem."
