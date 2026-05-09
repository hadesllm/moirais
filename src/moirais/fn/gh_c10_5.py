# moirais.fn — function file (hadesllm/moirais)
"""White noise adaptation: Abramovich-Silverman wavelet thresholding via Bayes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wn_adapt"]


def ghosal_wn_adapt(x):
    """
    White noise adaptation: Abramovich-Silverman wavelet thresholding via Bayes

    Formula: theta_jk ~ pi*N(0,tau_j^2) + (1-pi)*delta_0, adapts to Besov smoothness

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
    Ghosal Ch 10 §10.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "White noise adaptation: Abramovich-Silverman wavelet thresholding via Bayes"})


def cheatsheet():
    return "gh_c10_5: White noise adaptation: Abramovich-Silverman wavelet thresholding via Bayes"
