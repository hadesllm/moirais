"""Ensemble mean of a random process at instant t1 from M observations.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ensemble_mean"]


def rangayyan_ch3_ensemble_mean(x_k, t1, M):
    """
    Ensemble mean of a random process at instant t1 from M observations.

    Formula: mu_x(t1) = lim_{M->inf} (1/M) * sum_{k=1}^{M} x_k(t1)

    Parameters
    ----------
    x_k : array-like
        Input data.
    t1 : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.15, p. 96
    """
    x_k = np.atleast_1d(np.asarray(x_k, dtype=float))
    n = len(x_k)
    result = float(np.mean(x_k))
    se = float(np.std(x_k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ensemble mean of a random process at instant t1 from M observations."})


def cheatsheet():
    return "rng015: Ensemble mean of a random process at instant t1 from M observations."
