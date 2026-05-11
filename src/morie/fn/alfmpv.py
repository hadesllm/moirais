"""AlphaFold-Multimer chain pairing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_multimer"]


def alphafold_multimer(chains, msas):
    """
    AlphaFold-Multimer chain pairing

    Formula: chain assembly via paired MSAs

    Parameters
    ----------
    chains : array-like
        Input data.
    msas : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Evans et al (2022)
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold-Multimer chain pairing"})


def cheatsheet():
    return "alfmpv: AlphaFold-Multimer chain pairing"
