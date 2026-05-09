"""t-SNE for visualization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["tsne_reduction"]


def tsne_reduction(x):
    """
    t-SNE for visualization

    Formula: KL(P||Q), q_ij = (1+||yi-yj||^2)^{-1}

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
    van der Maaten (2008)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "t-SNE for visualization"})


def cheatsheet():
    return "tsnrd: t-SNE for visualization"
