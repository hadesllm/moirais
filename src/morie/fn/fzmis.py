# morie.fn -- function file (hadesllm/morie)
"""MISE decomposition for kernel density estimators (Fauzi Ch 1).

Asymptotic MISE of a 2nd-order kernel density estimator:

    MISE(h) = (h^4 / 4) * mu_2(K)^2 * R(f'')  +  R(K) / (n h).

The bandwidth minimising MISE is

    h_opt = ( R(K) / (n * mu_2(K)^2 * R(f'')) )^(1/5).

For Gaussian K, mu_2(K)=1, R(K)=1/(2 sqrt(pi)).  We estimate R(f'') by
the Silverman normal-reference plug-in R(f'') = 3 / (8 sqrt(pi) sigma^5).
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_mise_computation"]

_MU2_GAUSS = 1.0
_R_K_GAUSS = 1.0 / (2.0 * np.sqrt(np.pi))


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def fauzi_mise_computation(x, h=None):
    """Decomposed asymptotic MISE for a Gaussian KDE.

    Returns
    -------
    RichResult with estimate (MISE), bias_part, var_part, h, h_opt, n.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "fzmis -- too few obs"})

    sigma = float(np.std(x, ddof=1))
    if sigma <= 0:
        sigma = 1.0
    R_fpp = 3.0 / (8.0 * np.sqrt(np.pi) * sigma ** 5)

    if h is None:
        h = float(_silverman_h(x))

    bias_part = (h ** 4 / 4.0) * (_MU2_GAUSS ** 2) * R_fpp
    var_part = _R_K_GAUSS / (n * h)
    mise = bias_part + var_part

    h_opt = (_R_K_GAUSS / (n * (_MU2_GAUSS ** 2) * R_fpp)) ** (1.0 / 5.0)

    return RichResult(payload={
        "estimate": float(mise),
        "bias_part": float(bias_part),
        "var_part": float(var_part),
        "h": float(h),
        "h_opt": float(h_opt),
        "R_fpp": float(R_fpp),
        "sigma": float(sigma),
        "n": n,
        "method": "Fauzi MISE decomposition (Ch 1)",
    })


def cheatsheet():
    return "fzmis: MISE = c1*h^4 + c2/(nh); also returns h_opt"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_mise_computation(x)
# >>> r["bias_part"] > 0 and r["var_part"] > 0
# True
