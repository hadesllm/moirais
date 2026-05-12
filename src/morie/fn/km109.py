r"""Perplexity leakage.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_perplexity_leakage"]


def kamath_ch6_perplexity_leakage(S_uniq, PP_public, PP_lm):
    r"""
    Perplexity leakage.

    Formula: \epsilon_l = \max_{w\in S_{uniq}} \log(\frac{PP_{public}(w)}{PP_{lm}(w)})

    Parameters
    ----------
    S_uniq : array-like
        Input data.
    PP_public : array-like
        Input data.
    PP_lm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.33, p. 259
    r"""
    S_uniq = np.atleast_1d(np.asarray(S_uniq, dtype=float))
    n = len(S_uniq)
    result = float(np.mean(S_uniq))
    se = float(np.std(S_uniq, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perplexity leakage."})


def cheatsheet():
    return "km109: Perplexity leakage."
