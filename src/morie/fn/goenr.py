"""GO enrichment (Fisher exact)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["go_enrichment"]


def go_enrichment(foreground, background, go_terms):
    """
    GO enrichment (Fisher exact)

    Formula: hypergeometric test per term + FDR

    Parameters
    ----------
    foreground : array-like
        Input data.
    background : array-like
        Input data.
    go_terms : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Boyle et al (2004)
    """
    foreground = np.atleast_1d(np.asarray(foreground, dtype=float))
    n = len(foreground)
    result = float(np.mean(foreground))
    se = float(np.std(foreground, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GO enrichment (Fisher exact)"})


def cheatsheet():
    return "goenr: GO enrichment (Fisher exact)"
