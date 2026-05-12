r"""Bleu n geom mean.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_bleu_n_geom_mean"]


def kamath_ch8_bleu_n_geom_mean(p_n, N):
    r"""
    Bleu n geom mean.

    Formula: \mathrm{BLEU\text{-}N} = (\prod_{n=1}^N p_n)^{1/N}

    Parameters
    ----------
    p_n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.3, p. 323
    r"""
    p_n = np.atleast_1d(np.asarray(p_n, dtype=float))
    n = len(p_n)
    result = float(np.mean(p_n))
    se = float(np.std(p_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bleu n geom mean."})


def cheatsheet():
    return "km115: Bleu n geom mean."
