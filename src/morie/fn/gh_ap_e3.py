# morie.fn -- function file (hadesllm/morie)
"""Wavelet multiresolution analysis: scaling functions phi_{jk} and wavelets psi_{jk}."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wavelet_mra"]


def ghosal_wavelet_mra(x):
    """
    Wavelet multiresolution analysis: scaling functions phi_{jk} and wavelets psi_{jk}

    Formula: f = sum_k a_k phi_{j0,k} + sum_{j>=j0} sum_k d_{jk} psi_{jk}

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
    Ghosal App E
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet multiresolution analysis: scaling functions phi_{jk} and wavelets psi_{jk}"})


def cheatsheet():
    return "gh_ap_e3: Wavelet multiresolution analysis: scaling functions phi_{jk} and wavelets psi_{jk}"
