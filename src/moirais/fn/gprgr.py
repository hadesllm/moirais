# moirais.fn — function file (hadesllm/moirais)
"""Gaussian process regression with SE kernel. 'Rebellions are built on hope.' -- Jyn"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def gaussian_process_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray | None = None,
    length_scale: float = 1.0,
    output_scale: float = 1.0,
    noise_var: float = 0.1,
) -> dict:
    r"""
    Gaussian process regression with squared-exponential (RBF) kernel.

    Given training data :math:`(x_i, y_i)`, the GP makes predictions with:

    .. math::

        k(x, x') = \sigma^2 \exp\left( -\frac{\|x - x'\|^2}{2 \ell^2} \right) + \sigma_n^2 \delta_{xx'}

    The posterior mean and covariance at test points :math:`x_*` are:

    .. math::

        \bar{f}(x_*) = k(x_*, X) [K + \sigma_n^2 I]^{-1} \mathbf{y}

    :param x_train: (n,) or (n, d) training inputs.
    :type x_train: np.ndarray
    :param y_train: (n,) training outputs.
    :type y_train: np.ndarray
    :param x_test: (m,) or (m, d) test inputs. If None, uses x_train.
    :type x_test: np.ndarray | None
    :param length_scale: Kernel length scale (ell). Default 1.0.
    :type length_scale: float
    :param output_scale: Kernel output scale (sigma). Default 1.0.
    :type output_scale: float
    :param noise_var: Observation noise variance. Default 0.1.
    :type noise_var: float
    :return: Dictionary with keys: 'mean', 'var', 'x_test'.
    :rtype: dict

    References
    ----------
    Rasmussen C.E., Williams C.K.I. (2006). Gaussian Processes for Machine
    Learning. MIT Press.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)

    n = len(y_train)

    if x_test is None:
        x_test = x_train.copy()
    else:
        x_test = np.asarray(x_test, dtype=float)
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)

    m = len(x_test)

    # Squared-exponential kernel
    def se_kernel(x1, x2):
        sq_dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return output_scale**2 * np.exp(-sq_dist / (2 * length_scale**2))

    K_train = se_kernel(x_train, x_train)
    K_train += noise_var * np.eye(n)

    K_test = se_kernel(x_test, x_train)
    K_test_test = se_kernel(x_test, x_test)

    # Cholesky decomposition for stability
    try:
        L = cholesky(K_train, lower=True)
        alpha = solve_triangular(L, y_train, lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
    except np.linalg.LinAlgError:
        # Fallback to direct inversion
        K_inv = np.linalg.inv(K_train)
        alpha = K_inv @ y_train

    # Posterior mean
    mean = K_test @ alpha

    # Posterior covariance
    try:
        v = solve_triangular(L, K_test.T, lower=True)
        var = np.diag(K_test_test) - np.sum(v**2, axis=0)
    except:
        var = np.diag(K_test_test) - np.diag(K_test @ np.linalg.inv(K_train) @ K_test.T)

    return {
        "mean": mean,
        "var": var,
        "x_test": x_test,
        "length_scale": length_scale,
        "output_scale": output_scale,
        "noise_var": noise_var,
    }


gprgr = gaussian_process_regression


def cheatsheet() -> str:
    return "gaussian_process_regression(x_train, y_train, x_test) -> GP regression"
