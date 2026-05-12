"""Non-stationary covariance estimation (moving-window deformation-style)."""
import numpy as np
from scipy.spatial.distance import cdist
from ._richresult import RichResult

__all__ = ["nonstationary_covariance"]


def nonstationary_covariance(x, coords, bandwidth: float | None = None):
    """
    Non-stationary covariance estimator C(s1, s2) = sigma(s1) sigma(s2) rho(s1, s2),
    with a moving-window decomposition:

        sigma_hat(s_i) = sqrt(  sum_j K_b(s_i - s_j) (x_j - mu_hat(s_i))^2
                              / sum_j K_b(s_i - s_j) )
        rho_hat(s_i, s_j) = (x_i - mu_hat(s_i)) (x_j - mu_hat(s_j))
                            / (sigma_hat(s_i) sigma_hat(s_j))
        C_hat(s_i, s_j) = sigma_hat(s_i) sigma_hat(s_j) rho_hat(s_i, s_j).

    Local means and SDs are kernel-weighted; the kernel is Gaussian with
    bandwidth `bandwidth` (default: median pairwise distance).

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    bandwidth : float, optional

    Returns
    -------
    RichResult with payload:
        estimate : dict {sigma_local, C_matrix}
        n, method
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = x.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    D = cdist(coords, coords)
    if bandwidth is None:
        bandwidth = float(np.median(D[D > 0])) if n > 1 else 1.0
    if bandwidth <= 0:
        bandwidth = 1.0
    K = np.exp(-0.5 * (D / bandwidth) ** 2)
    wsum = K.sum(axis=1)
    mu_local = (K @ x) / np.where(wsum > 0, wsum, 1.0)
    dev = x - mu_local
    var_local = (K @ (dev ** 2)) / np.where(wsum > 0, wsum, 1.0)
    sigma_local = np.sqrt(np.maximum(var_local, 1e-12))
    # Standardised residuals
    eps = dev / sigma_local
    # Empirical correlation of standardised residuals smoothed by K
    rho = K * np.outer(eps, eps) / (np.sqrt(np.outer(wsum, wsum)) + 1e-12)
    C = np.outer(sigma_local, sigma_local) * rho
    return RichResult(payload={
        "estimate": {
            "sigma_local": sigma_local.tolist(),
            "C_matrix": C.tolist(),
            "bandwidth": float(bandwidth),
        },
        "n": int(n),
        "method": "Non-stationary covariance (moving-window kernel)",
    })


def cheatsheet():
    return "nstat: Non-stationary covariance estimation"


# CANONICAL TEST
# x = [1,2,3,4,5], coords = [[0],[1],[2],[3],[4]]
# sigma_local positive, diag(C_matrix) > 0
