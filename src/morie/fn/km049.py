r"""Top1 prompt metric.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_top1_prompt_metric"]


def kamath_ch3_top1_prompt_metric(R, t, P_LM):
    r"""
    Top1 prompt metric.

    Formula: A(t_{r,i}) = \frac{\sum_{(x,y)\in R}\delta(y = \arg\max_{y'} P_{LM}(y'|x,t_{r,i}))}{|R|}

    Parameters
    ----------
    R : array-like
        Input data.
    t : array-like
        Input data.
    P_LM : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.8, p. 102
    r"""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Top1 prompt metric."})


def cheatsheet():
    return "km049: Top1 prompt metric."
