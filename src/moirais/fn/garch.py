# moirais.fn — function file (hadesllm/moirais)
"""GARCH(1,1) volatility model."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def garch_fit(returns: np.ndarray) -> DescriptiveResult:
    """
    Fit a GARCH(1,1) model to financial returns.

    .. math::

        \\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2

    Estimated via MLE with Gaussian innovations.

    :param returns: (n,) array of returns (demeaned recommended).
    :return: DescriptiveResult with omega, alpha, beta and conditional variances.
    :raises ValueError: If series too short.

    References
    ----------
    Bollerslev T (1986). Generalized autoregressive conditional
    heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r - r.mean()
    n = len(r)
    if n < 10:
        raise ValueError("Need at least 10 observations.")

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else np.var(r)
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-10)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r**2 / sigma2)
        return -ll

    var_r = float(np.var(r))
    x0 = [var_r * 0.05, 0.1, 0.85]
    bounds = [(1e-8, var_r * 10), (1e-8, 0.999), (1e-8, 0.999)]
    res = optimize.minimize(neg_loglik, x0, bounds=bounds, method="L-BFGS-B")
    omega, alpha, beta = res.x
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else var_r
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    return DescriptiveResult(
        name="garch",
        value=float(omega / (1 - alpha - beta)) if (1 - alpha - beta) > 0 else float(var_r),
        extra={
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistence": float(alpha + beta),
            "conditional_variance": sigma2,
            "loglik": float(-res.fun),
            "n": n,
        },
    )


garch = garch_fit


def cheatsheet() -> str:
    return "garch_fit({}) -> GARCH(1,1) volatility model."
