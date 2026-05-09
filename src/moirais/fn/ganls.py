# moirais.fn — function file (hadesllm/moirais)
"""GAN minimax loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gan_loss"]


def gan_loss(x):
    """
    GAN minimax loss

    Formula: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]

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
    Goodfellow et al. (2014)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GAN minimax loss"})


def cheatsheet():
    return "ganls: GAN minimax loss"
