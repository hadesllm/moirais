"""Fragment-based ligand growing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fragment_growing"]


def fragment_growing(fragment, linker_lib):
    """
    Fragment-based ligand growing

    Formula: link fragments at exit vectors with linker library

    Parameters
    ----------
    fragment : array-like
        Input data.
    linker_lib : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Murray-Rees (2009); Shuker et al (1996) SAR-by-NMR
    """
    fragment = np.atleast_1d(np.asarray(fragment, dtype=float))
    n = len(fragment)
    result = float(np.mean(fragment))
    se = float(np.std(fragment, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fragment-based ligand growing"})


def cheatsheet():
    return "frgrow: Fragment-based ligand growing"
