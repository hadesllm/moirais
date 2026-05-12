r"""Reward loss pairwise.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_reward_loss_pairwise"]


def kamath_ch5_reward_loss_pairwise(r_theta, x, y_0, y_1, i):
    r"""
    Reward loss pairwise.

    Formula: \mathrm{loss}(r_{\theta}) = -E_{(x,y_0,y_1,i)\sim D}[\log(\sigma(r_{\theta}(x,y_i) - r_{\theta}(x,y_{1-i})))]

    Parameters
    ----------
    r_theta : array-like
        Input data.
    x : array-like
        Input data.
    y_0 : array-like
        Input data.
    y_1 : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.1, p. 195
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reward loss pairwise."})


def cheatsheet():
    return "km065: Reward loss pairwise."
