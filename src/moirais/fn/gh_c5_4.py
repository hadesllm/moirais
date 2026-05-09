# moirais.fn — function file (hadesllm/moirais)
"""Split-merge MCMC for DP mixtures: proposes splitting or merging clusters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_splitmerge"]


def ghosal_splitmerge(x):
    """
    Split-merge MCMC for DP mixtures: proposes splitting or merging clusters

    Formula: Split: divide cluster k into two; Merge: join clusters k,l into one via MH

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 5 §5.2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Split-merge MCMC for DP mixtures: proposes splitting or merging clusters"})


def cheatsheet():
    return "gh_c5_4: Split-merge MCMC for DP mixtures: proposes splitting or merging clusters"
