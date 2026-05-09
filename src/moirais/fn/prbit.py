# moirais.fn — function file (hadesllm/moirais)
"""Ordinal probit coefficients for cutting-plane normal vectors."""

from __future__ import annotations

from ._containers import DescriptiveResult


def ordinal_probit_coefficients(Y, X, cdf=None) -> DescriptiveResult:
    """Estimate ordinal probit coefficients via maximum likelihood.

    Uses a simple binary probit (threshold at median) as fast approximation.

    :param Y: Ordinal response vector.
    :param X: Predictor matrix (ideal points).
    :return: DescriptiveResult with coefficient estimates.

    .. epigraph:: "Serious Series: Serious Punch." -- Saitama, One Punch Man
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import norm

    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y_bin = (np.median(Y) < Y).astype(float)

    def neg_ll(beta):
        eta = X @ beta
        p = norm.cdf(eta)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y_bin * np.log(p) + (1 - y_bin) * np.log(1 - p))

    b0 = np.zeros(X.shape[1])
    res = minimize(neg_ll, b0, method="BFGS")
    return DescriptiveResult(
        name="ordinal_probit_coefficients",
        value=float(res.fun),
        extra={"coefficients": res.x.tolist(), "converged": bool(res.success)},
    )


prbit = ordinal_probit_coefficients


def cheatsheet() -> str:
    return "ordinal_probit_coefficients({}) -> Ordinal probit coefficients for cutting-plane normal vectors"
