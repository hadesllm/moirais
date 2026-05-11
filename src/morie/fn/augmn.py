# morie.fn — function file (hadesllm/morie)
"""Albert-Chib data augmentation for binary ordinal Gibbs sampler."""
import numpy as np
from ._richresult import RichResult

__all__ = ["albert_chib_augmentation"]


def albert_chib_augmentation(y_bin, X, Z):
    """
    Albert-Chib data augmentation for binary ordinal Gibbs sampler

    Formula: z_i ~ TN(eta_i, 1, [tau_{y_i-1}, tau_{y_i}]); sample latent z, then regress z on predictors

    Parameters
    ----------
    y_bin : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'z_samples': 'array', 'beta_samples': 'array'}

    References
    ----------
    Montesinos Lopez Ch 7
    """
    y_bin = np.asarray(y_bin, dtype=float)
    n = int(y_bin) if y_bin.ndim == 0 else len(y_bin)
    result = float(np.mean(y_bin))
    se = float(np.std(y_bin, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Albert-Chib data augmentation for binary ordinal Gibbs sampler"})


def cheatsheet():
    return "augmn: Albert-Chib data augmentation for binary ordinal Gibbs sampler"
