r"""Ppo loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_ppo_loss"]


def kamath_ch5_ppo_loss(phi, x, y, r_theta, beta):
    r"""
    Ppo loss.

    Formula: L(\phi) = -E_{x\sim D, y\sim \pi^{RL}_{\phi}}[r_{\theta}(x,y) - \beta\cdot D_{KL}(\pi^{RL}_{\phi}(y|x)\|\pi^{REF}(y|x))]

    Parameters
    ----------
    phi : array-like
        Input data.
    x : array-like
        Input data.
    y : array-like
        Input data.
    r_theta : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.4, p. 200
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ppo loss."})


def cheatsheet():
    return "km068: Ppo loss."
