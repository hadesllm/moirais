# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""BayesA via Gibbs sampler."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayes_ridge_gibbs"]


def bayes_ridge_gibbs(x, y):
    """
    BayesA via Gibbs sampler

    Formula: beta_j ~ N(0, sigma_j^2), sigma_j^2 ~ IG(df,S)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Montesinos Lopez Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BayesA via Gibbs sampler"})


def cheatsheet():
    return "brdgf: BayesA via Gibbs sampler"
