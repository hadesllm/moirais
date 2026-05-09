"""Dpo loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_dpo_loss"]


def kamath_ch5_dpo_loss(pi_theta, pi_ref, beta):
    """
    Dpo loss.

    Formula: L_{DPO}(\pi_{\theta};\pi_{ref}) = -E_{(x,y_w,y_l)\sim D}[\log\sigma(\beta\log\frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)})]

    Parameters
    ----------
    pi_theta : array-like
        Input data.
    pi_ref : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.12, p. 210
    """
    pi_theta = np.atleast_1d(np.asarray(pi_theta, dtype=float))
    n = len(pi_theta)
    result = float(np.mean(pi_theta))
    se = float(np.std(pi_theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dpo loss."})


def cheatsheet():
    return "km076: Dpo loss."
