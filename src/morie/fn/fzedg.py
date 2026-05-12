# morie.fn — function file (hadesllm/morie)
"""Edgeworth expansion for kernel quantile (Fauzi Ch 3).

One-term Edgeworth correction to the normal approximation of the
studentised kernel quantile statistic T_n = sqrt(n) (Q_hat - Q) / sigma:

    P( T_n <= z ) ≈ Phi(z) + n^{-1/2} * p1(z) * phi(z)

where p1(z) is a polynomial in z whose coefficients depend on the
skewness/kurtosis of the kernel-CDF score function.  For symmetric
kernels and central-quantile p, p1(z) = -(skew/6) * (z^2 - 1), which
follows from the Cornish-Fisher inverse-Edgeworth.

This function returns the Edgeworth-corrected tail probability and the
Cornish-Fisher correction to the Gaussian critical value.
"""
import numpy as np
from scipy import stats as _sps
from ._richresult import RichResult

__all__ = ["fauzi_edgeworth_quantile"]


def fauzi_edgeworth_quantile(x, z=1.96, p=0.5):
    """Edgeworth-corrected tail probability for kernel-quantile T_n.

    Parameters
    ----------
    x : array-like
    z : float    standardised critical value (default 1.96)
    p : float    quantile probability level (default 0.5)
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "fzedg — too few obs"})

    # Indicator I{X<=Q(p)} has variance p(1-p), third central moment
    # p(1-p)(1-2p), so standardised skewness γ1 = (1-2p)/sqrt(p(1-p)).
    skew = (1.0 - 2.0 * p) / np.sqrt(p * (1.0 - p))

    p1z = -(skew / 6.0) * (z * z - 1.0)
    phi_z = _sps.norm.pdf(z)
    Phi_z = _sps.norm.cdf(z)
    correction = p1z * phi_z / np.sqrt(n)
    corrected = float(Phi_z + correction)

    cf_correction = (skew / 6.0) * (z * z - 1.0) / np.sqrt(n)

    return RichResult(payload={
        "estimate": corrected,
        "normal_approx": float(Phi_z),
        "edgeworth_correction": float(correction),
        "cornish_fisher_correction": float(cf_correction),
        "skew": float(skew),
        "p1z": float(p1z),
        "z": z,
        "p": p,
        "n": n,
        "method": "Fauzi Edgeworth expansion for kernel quantile (Ch 3)",
    })


def cheatsheet():
    return "fzedg: Edgeworth + Cornish-Fisher correction for kernel quantile"


# CANONICAL TEST
# >>> r = fauzi_edgeworth_quantile([1,2,3,4,5,6,7,8,9,10], z=1.96, p=0.5)
# >>> abs(r["skew"]) < 1e-12  # symmetric quantile, no skew term
# True
