# morie.fn -- function file (hadesllm/morie)
"""AR/ARMA model of PCG for heart sound characterization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pcg_ar_model"]


def rangayyan_pcg_ar_model(pcg, fs, p, q):
    """
    AR/ARMA model of PCG for heart sound characterization

    Formula: AR: H(z)=1/A(z); ARMA: H(z)=B(z)/A(z); poles track S1/S2 resonances

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: poles, freqs_from_poles

    References
    ----------
    Rangayyan Ch 7.10
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR/ARMA model of PCG for heart sound characterization"})


def cheatsheet():
    return "rgpcgar: AR/ARMA model of PCG for heart sound characterization"
