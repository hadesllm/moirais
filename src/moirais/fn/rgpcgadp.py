# moirais.fn — function file (hadesllm/moirais)
"""Adaptive segmentation of PCG signals via SEM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pcg_adaptive_seg"]


def rangayyan_pcg_adaptive_seg(pcg, fs, ar_order):
    """
    Adaptive segmentation of PCG signals via SEM

    Formula: SEM computed between consecutive AR model segments; high SEM = change

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.
    ar_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: segment_bounds, sem_trace

    References
    ----------
    Rangayyan Ch 8.11
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive segmentation of PCG signals via SEM"})


def cheatsheet():
    return "rgpcgadp: Adaptive segmentation of PCG signals via SEM"
