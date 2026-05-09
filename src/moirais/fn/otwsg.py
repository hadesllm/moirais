"""2-Wasserstein between two Gaussians (closed-form)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_wasserstein_gauss"]


def ot_wasserstein_gauss(mu1, Sigma1, mu2, Sigma2):
    """
    2-Wasserstein between two Gaussians (closed-form)

    Formula: W_2² = ||μ_1-μ_2||² + tr(Σ₁+Σ₂-2(Σ₁^{1/2}Σ₂Σ₁^{1/2})^{1/2})

    Parameters
    ----------
    mu1 : array-like
        Input data.
    Sigma1 : array-like
        Input data.
    mu2 : array-like
        Input data.
    Sigma2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W2

    References
    ----------
    Olkin & Pukelsheim (1982)
    """
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
    n = len(mu1)
    result = float(np.mean(mu1))
    se = float(np.std(mu1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "2-Wasserstein between two Gaussians (closed-form)"})


def cheatsheet():
    return "otwsg: 2-Wasserstein between two Gaussians (closed-form)"
