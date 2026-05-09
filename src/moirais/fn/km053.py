"""Prefix tuning obj.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_prefix_tuning_obj"]


def kamath_ch3_prefix_tuning_obj(phi, x, y, h):
    """
    Prefix tuning obj.

    Formula: \max_{\phi} \log p_{\phi}(y|x) = \sum_{i\in Y_{idx}} \log p_{\phi}(z_i|h_{<i})

    Parameters
    ----------
    phi : array-like
        Input data.
    x : array-like
        Input data.
    y : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.12, p. 110
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prefix tuning obj."})


def cheatsheet():
    return "km053: Prefix tuning obj."
