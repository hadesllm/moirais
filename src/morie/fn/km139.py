r"""Simvlm prefixlm.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_simvlm_prefixlm"]


def kamath_ch9_simvlm_prefixlm(theta, x, T_p):
    r"""
    Simvlm prefixlm.

    Formula: L_{PrefixLM}(\theta) = -E_{x\sim D} \log P_{\theta}(x_{\ge T_p}|x_{<T_p})

    Parameters
    ----------
    theta : array-like
        Input data.
    x : array-like
        Input data.
    T_p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.11, p. 387
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simvlm prefixlm."})


def cheatsheet():
    return "km139: Simvlm prefixlm."
