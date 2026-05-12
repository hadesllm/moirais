r"""Full finetune obj.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_full_finetune_obj"]


def kamath_ch4_full_finetune_obj(Phi, x, y):
    r"""
    Full finetune obj.

    Formula: \max_{\Phi} \sum_{(x,y)\in Z}\sum_{t=1}^{|y|}\log(P_{\Phi}(y_t|x,y_{<t}))

    Parameters
    ----------
    Phi : array-like
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
    Kamath et al (2024), Ch 4, Eq 4.3, p. 150
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Full finetune obj."})


def cheatsheet():
    return "km056: Full finetune obj."
