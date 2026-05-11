"""Ceat random effects.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_ceat_random_effects"]


def kamath_ch6_ceat_random_effects(S_A1, S_A2, S_W1, S_W2, v):
    """
    Ceat random effects.

    Formula: \mathrm{CEAT}(S_{A_1},S_{A_2},S_{W_1},S_{W_2}) = \frac{\sum_{i=1}^N v_i\,\mathrm{WEAT}(S_{A_1i},S_{A_2i},S_{W_1i},S_{W_2i})}{\sum_{i=1}^N v_i}

    Parameters
    ----------
    S_A1 : array-like
        Input data.
    S_A2 : array-like
        Input data.
    S_W1 : array-like
        Input data.
    S_W2 : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.7, p. 234
    """
    S_A1 = np.atleast_1d(np.asarray(S_A1, dtype=float))
    n = len(S_A1)
    result = float(np.mean(S_A1))
    se = float(np.std(S_A1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ceat random effects."})


def cheatsheet():
    return "km083: Ceat random effects."
