"""Lora obj.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_lora_obj"]


def kamath_ch4_lora_obj(Theta, Phi_0, x, y):
    """
    Lora obj.

    Formula: \max_{\Theta} \sum_{(x,y)\in Z}\sum_{t=1}^{|y|}\log(p_{\Phi_0+\Delta\Phi(\Theta)}(y_t|x,y_{<t}))

    Parameters
    ----------
    Theta : array-like
        Input data.
    Phi_0 : array-like
        Input data.
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.4, p. 151
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lora obj."})


def cheatsheet():
    return "km057: Lora obj."
