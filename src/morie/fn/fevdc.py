# morie.fn — function file (hadesllm/morie)
"""Forecast error variance decomposition from a VAR(1) model."""

import numpy as np

from ._containers import DescriptiveResult


def fevd(
    var_coefficients: np.ndarray,
    sigma_u: np.ndarray,
    periods: int = 20,
) -> DescriptiveResult:
    """
    Forecast error variance decomposition from a VAR(1).

    Computes the fraction of forecast error variance of each variable
    attributable to shocks in each variable, using the Cholesky
    decomposition of the residual covariance.

    :param var_coefficients: VAR(1) coefficient matrix A (k x k).
    :param sigma_u: Residual covariance matrix (k x k).
    :param periods: Forecast horizon. Default 20.
    :return: DescriptiveResult with decomposition array (periods, k, k).
    :raises ValueError: If matrices are not conformable.

    References
    ----------
    Lutkepohl H. (2005). New Introduction to Multiple Time Series
    Analysis. Springer.
    """
    A = np.asarray(var_coefficients, dtype=float)
    S = np.asarray(sigma_u, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"var_coefficients must be square, got {A.shape}.")
    k = A.shape[0]
    if S.shape != (k, k):
        raise ValueError(f"sigma_u must be ({k},{k}), got {S.shape}.")
    P = np.linalg.cholesky(S)
    Theta = np.zeros((periods + 1, k, k))
    Theta[0] = np.eye(k)
    A_pow = np.eye(k)
    for h in range(1, periods + 1):
        A_pow = A_pow @ A
        Theta[h] = A_pow
    mse = np.zeros((periods + 1, k, k))
    decomp = np.zeros((periods + 1, k, k))
    for h in range(periods + 1):
        contrib = np.zeros((k, k))
        for s in range(h + 1):
            ThetaP = Theta[s] @ P
            contrib += ThetaP ** 2
        total_var = contrib.sum(axis=1)
        for j in range(k):
            for i in range(k):
                decomp[h, i, j] = contrib[i, j] / total_var[i] if total_var[i] > 0 else 0
        mse[h] = contrib
    return DescriptiveResult(
        name="fevd",
        value=float(decomp[-1, 0, 0]),
        extra={
            "decomposition": decomp,
            "mse_contributions": mse,
            "periods": periods,
            "k": k,
        },
    )


fevdc = fevd


def cheatsheet() -> str:
    return "fevd({}) -> Forecast error variance decomposition."
