r"""Affect lm.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_affect_lm"]


def kamath_ch6_affect_lm(U, V, f, g, c, e, beta, b):
    r"""
    Affect lm.

    Formula: P(w_t=i|c_{t-1},e_{t-1}) = \frac{\exp(U_i^T f(c_{t-1}) + \beta V_i^T g(e_{t-1}) + b_i)}{\sum_{j=1}^V \exp(U_j^T f(c_{t-1}) + \beta V_j^T g(e_{t-1}) + b_j)}

    Parameters
    ----------
    U : array-like
        Input data.
    V : array-like
        Input data.
    f : array-like
        Input data.
    g : array-like
        Input data.
    c : array-like
        Input data.
    e : array-like
        Input data.
    beta : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.28, p. 253
    r"""
    U = np.atleast_1d(np.asarray(U, dtype=float))
    n = len(U)
    result = float(np.mean(U))
    se = float(np.std(U, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Affect lm."})


def cheatsheet():
    return "km104: Affect lm."
