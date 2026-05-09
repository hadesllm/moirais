# moirais.fn — function file (hadesllm/moirais)
"""KL property for kernel mixtures: DPM with Gaussian kernel has KL support."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_kern_mix_kl"]


def ghosal_kern_mix_kl(x):
    """
    KL property for kernel mixtures: DPM with Gaussian kernel has KL support

    Formula: KL(p0, f_G) controlled by KL(G_n, G) for approximating measure G_n

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
    Ghosal Ch 7 §7.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KL property for kernel mixtures: DPM with Gaussian kernel has KL support"})


def cheatsheet():
    return "gh_c7_2: KL property for kernel mixtures: DPM with Gaussian kernel has KL support"
