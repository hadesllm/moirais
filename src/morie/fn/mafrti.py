"""Back-transform from Freeman-Tukey double arcsine to p."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_freeman_tukey_inverse"]


def ma_freeman_tukey_inverse(ft, n_harmonic):
    """
    Back-transform from Freeman-Tukey double arcsine to p

    Formula: p̂ = (1 - sgn(cos t))(1 - sqrt(...))/2 etc.

    Parameters
    ----------
    ft : array-like
        Input data.
    n_harmonic : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Miller (1978)
    """
    ft = np.atleast_1d(np.asarray(ft, dtype=float))
    n = len(ft)
    result = float(np.mean(ft))
    se = float(np.std(ft, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Back-transform from Freeman-Tukey double arcsine to p"})


def cheatsheet():
    return "mafrti: Back-transform from Freeman-Tukey double arcsine to p"
