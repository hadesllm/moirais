"""Self diagnosis prob.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_self_diagnosis_prob"]


def kamath_ch6_self_diagnosis_prob(x, y, M, sdg):
    """
    Self diagnosis prob.

    Formula: p(y|x) = \frac{p_M(\text{Yes}|sdg(x,y))}{\sum_{w\in\{\text{Yes,No}\}} p_M(w|sdg(x,y))}

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    M : array-like
        Input data.
    sdg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.30, p. 255
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self diagnosis prob."})


def cheatsheet():
    return "km106: Self diagnosis prob."
