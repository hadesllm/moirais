"""Ldm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_ldm_loss"]


def kamath_ch9_ldm_loss(epsilon, z_t, H_X):
    """
    Ldm loss.

    Formula: L_{X-gen} := E_{\epsilon\sim N(0,1), t} \|\epsilon - \epsilon_X(z_t, t, H_X)\|_2^2

    Parameters
    ----------
    epsilon : array-like
        Input data.
    z_t : array-like
        Input data.
    H_X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.20, p. 398
    """
    epsilon = np.atleast_1d(np.asarray(epsilon, dtype=float))
    n = len(epsilon)
    result = float(np.mean(epsilon))
    se = float(np.std(epsilon, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ldm loss."})


def cheatsheet():
    return "km148: Ldm loss."
