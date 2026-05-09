"""Heart rate computed from number of QRS complexes detected over duration T.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_heart_rate_from_count"]


def rangayyan_ch4_heart_rate_from_count(N_B, T):
    """
    Heart rate computed from number of QRS complexes detected over duration T.

    Formula: HR = 60 * N_B / T

    Parameters
    ----------
    N_B : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.19, p. 224
    """
    N_B = np.atleast_1d(np.asarray(N_B, dtype=float))
    n = len(N_B)
    result = float(np.mean(N_B))
    se = float(np.std(N_B, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heart rate computed from number of QRS complexes detected over duration T."})


def cheatsheet():
    return "rng193: Heart rate computed from number of QRS complexes detected over duration T."
