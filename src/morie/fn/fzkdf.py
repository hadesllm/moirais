# morie.fn — function file (hadesllm/morie)
"""KDFE bias and variance properties (Fauzi Ch 2).

Kernel distribution function estimator (KDFE):

    F_hat_h(t) = (1/n) * sum_i W((t - X_i)/h)

where W(u) = integral_{-inf}^{u} K(v) dv is the integrated kernel.
For Gaussian K, W = Phi.

Asymptotic bias and variance of the KDFE (Fauzi Ch 2, Eqn 2.x):

    Bias[F_hat_h(t)] = (h^2 / 2) * mu_2(K) * f'(t) + o(h^2)
    Var [F_hat_h(t)] = F(t)(1-F(t))/n - (2*h*r(K)/n) * f(t) + o(h/n)

with mu_2(K) = integral u^2 K(u) du, r(K) = integral u K(u) W(u) du.
For Gaussian K: mu_2 = 1, r(K) = 1/(2*sqrt(pi)).
"""
import numpy as np
from scipy import stats as _sps
from ._richresult import RichResult

__all__ = ["fauzi_kdfe_properties"]

# Gaussian kernel moments (analytic)
_MU2_GAUSS = 1.0
_RK_GAUSS = 1.0 / (2.0 * np.sqrt(np.pi))


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _kdfe(x, t, h):
    """Gaussian integrated-kernel CDF estimate at t."""
    z = (t - x) / h
    return float(np.mean(_sps.norm.cdf(z)))


def fauzi_kdfe_properties(x, t=None, h=None):
    """KDFE asymptotic bias and variance at evaluation point ``t``.

    Parameters
    ----------
    x : array-like
        Sample.
    t : float, optional
        Evaluation point.  Default = sample median.
    h : float, optional
        Bandwidth.  Default = Silverman's rule.

    Returns
    -------
    RichResult
        Keys: estimate (F_hat_h(t)), bias, variance, se, h, n.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "KDFE — too few obs"})
    if t is None:
        t = float(np.median(x))
    if h is None:
        h = float(_silverman_h(x))

    F_hat = _kdfe(x, t, h)
    # f'(t) approximated by a pilot Gaussian-KDE-based derivative
    z = (t - x) / h
    f_hat = float(np.mean(_sps.norm.pdf(z) / h))
    # f'(t) via the analytic derivative of the Gaussian-KDE
    fp_hat = float(np.mean(-z * _sps.norm.pdf(z) / (h * h)))

    bias = (h * h / 2.0) * _MU2_GAUSS * fp_hat
    var = F_hat * (1.0 - F_hat) / n - (2.0 * h * _RK_GAUSS / n) * f_hat
    var = max(var, 0.0)
    se = np.sqrt(var)

    return RichResult(payload={
        "estimate": F_hat,
        "bias": bias,
        "variance": var,
        "se": se,
        "h": h,
        "t": t,
        "n": n,
        "method": "Fauzi KDFE bias-variance (Ch 2)",
    })


def cheatsheet():
    return "fzkdf: KDFE asymptotic bias and variance at point t"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_kdfe_properties(x, t=0.0)
# >>> 0.4 < r["estimate"] < 0.6   # F(0) ≈ 0.5 for standard normal
# True
