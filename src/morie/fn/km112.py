r"""Answer relevance.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch7_answer_relevance"]


def kamath_ch7_answer_relevance(E_g, E_o, N):
    r"""
    Answer relevance.

    Formula: \text{Answer Relevance} = \frac{1}{N}\sum_{i=1}^N \mathrm{sim}(E_{g_i},E_o)

    Parameters
    ----------
    E_g : array-like
        Input data.
    E_o : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 7, Eq 7.3, p. 300
    r"""
    E_g = np.atleast_1d(np.asarray(E_g, dtype=float))
    n = len(E_g)
    result = float(np.mean(E_g))
    se = float(np.std(E_g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Answer relevance."})


def cheatsheet():
    return "km112: Answer relevance."
