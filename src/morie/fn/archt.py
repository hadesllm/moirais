# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""ARCH(p) conditional heteroscedasticity model."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def arch_fit(returns: np.ndarray, p: int = 1) -> DescriptiveResult:
    r"""
    Fit an ARCH(p) model to returns via MLE.

    .. math::

        \\sigma_t^2 = \\omega + \\sum_{i=1}^{p} \\alpha_i \\varepsilon_{t-i}^2

    :param returns: 1-D array of returns (demeaned recommended).
    :param p: ARCH order. Default 1.
    :return: DescriptiveResult with omega, alpha, conditional variances.
    :raises ValueError: If series too short.

    References
    ----------
    Engle R.F. (1982). Autoregressive conditional heteroscedasticity
    with estimates of the variance of UK inflation. *Econometrica*,
    50(4), 987-1007.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r - r.mean()
    n = len(r)
    if n < p + 10:
        raise ValueError(f"Need at least {p + 10} observations for ARCH({p}), got {n}.")

    def neg_loglik(params):
        omega = params[0]
        alpha = params[1:]
        if omega <= 0 or np.any(alpha < 0) or np.sum(alpha) >= 1:
            return 1e10
        sigma2 = np.full(n, omega / (1 - np.sum(alpha)) if np.sum(alpha) < 1 else np.var(r))
        for t in range(p, n):
            sigma2[t] = omega + sum(alpha[i] * r[t - 1 - i] ** 2 for i in range(p))
            sigma2[t] = max(sigma2[t], 1e-10)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2[p:]) + r[p:] ** 2 / sigma2[p:])
        return -ll

    var_r = float(np.var(r))
    x0 = [var_r * 0.1] + [0.1 / p] * p
    bounds = [(1e-8, var_r * 10)] + [(1e-8, 0.999)] * p
    res = optimize.minimize(neg_loglik, x0, bounds=bounds, method="L-BFGS-B")
    omega = float(res.x[0])
    alpha = res.x[1:].tolist()
    sigma2 = np.full(n, omega / (1 - sum(alpha)) if sum(alpha) < 1 else var_r)
    for t in range(p, n):
        sigma2[t] = omega + sum(alpha[i] * r[t - 1 - i] ** 2 for i in range(p))
    return DescriptiveResult(
        name="arch_fit",
        value=float(omega / (1 - sum(alpha))) if sum(alpha) < 1 else var_r,
        extra={
            "omega": omega,
            "alpha": alpha,
            "persistence": sum(alpha),
            "conditional_variance": sigma2,
            "loglik": float(-res.fun),
            "p": p,
            "n": n,
        },
    )


archt = arch_fit


def cheatsheet() -> str:
    return "arch_fit({}) -> ARCH(p) conditional heteroscedasticity model."
