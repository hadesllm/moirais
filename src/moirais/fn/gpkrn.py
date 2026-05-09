# moirais.fn — function file (hadesllm/moirais)
"""GP kernel functions: SE, Matern, RQ, periodic. 'These are the droids.' -- ObiWan"""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def squared_exponential_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, output_scale: float = 1.0
) -> np.ndarray:
    r"""
    Squared-exponential (RBF) kernel.

    .. math::
        k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    sq_dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
    return output_scale**2 * np.exp(-sq_dist / (2 * length_scale**2))


def matern_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, output_scale: float = 1.0, nu: float = 1.5
) -> np.ndarray:
    r"""Matern kernel; interpolates between exponential (nu=0.5) and SE (nu->inf)."""
    from scipy.special import gamma, kv

    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    dist = np.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=2)
    scaled_dist = np.sqrt(2.0 * nu) * dist / length_scale + 1e-8

    coeff = (2.0 ** (1.0 - nu)) / gamma(nu)
    return output_scale**2 * coeff * (scaled_dist ** nu) * kv(nu, scaled_dist)


def rational_quadratic_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, output_scale: float = 1.0, alpha: float = 1.0
) -> np.ndarray:
    r"""Rational quadratic kernel: mixture of SE kernels at different scales."""
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    sq_dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
    return output_scale**2 * (1.0 + sq_dist / (2.0 * alpha * length_scale**2)) ** (-alpha)


def periodic_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, output_scale: float = 1.0, period: float = 1.0
) -> np.ndarray:
    r"""Periodic kernel: captures periodic structure."""
    x1 = np.atleast_1d(x1).ravel()
    x2 = np.atleast_1d(x2).ravel()

    dist = np.abs(x1[:, None] - x2[None, :])
    sin_dist = np.sin(np.pi * dist / period)
    return output_scale**2 * np.exp(-2.0 * sin_dist**2 / length_scale**2)


def gp_kernel_matrix(
    x1: np.ndarray, x2: np.ndarray, kernel_type: str = 'se', **kwargs
) -> np.ndarray:
    """Compute a GP kernel matrix using the requested kernel family."""
    if kernel_type == 'se':
        return squared_exponential_kernel(x1, x2, **{k: v for k, v in kwargs.items() if k in ['length_scale', 'output_scale']})
    if kernel_type == 'matern':
        return matern_kernel(x1, x2, **{k: v for k, v in kwargs.items() if k in ['length_scale', 'output_scale', 'nu']})
    if kernel_type == 'rq':
        return rational_quadratic_kernel(x1, x2, **{k: v for k, v in kwargs.items() if k in ['length_scale', 'output_scale', 'alpha']})
    if kernel_type == 'periodic':
        return periodic_kernel(x1, x2, **{k: v for k, v in kwargs.items() if k in ['length_scale', 'output_scale', 'period']})
    raise ValueError(f"Unknown kernel type: {kernel_type}")


def gpkrn(
    data: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    n: int = 50,
    seed: int = 0,
    kernel_type: str = 'se',
    **kwargs,
) -> SpatialResult:
    """
    Build a GP kernel matrix and return a SpatialResult summary.

    :param data: (n,) observations. Synthesized from RNG when None.
    :param coords: (n, d) coordinates. Synthesized uniformly when None.
    :param n: Sample size when synthesizing.
    :param seed: RNG seed.
    :param kernel_type: 'se', 'matern', 'rq', or 'periodic'.
    :return: SpatialResult with statistic=mean(data) and kernel summary in .extra.
    """
    rng = np.random.default_rng(seed)
    if data is None:
        data = rng.standard_normal(n)
    data = np.asarray(data, dtype=float).ravel()
    n_points = len(data)
    if coords is None:
        coords = rng.uniform(0, 1, size=(n_points, 2))
    coords = np.asarray(coords, dtype=float)

    kernel_kwargs = {k: v for k, v in kwargs.items() if k in {'length_scale', 'output_scale', 'nu', 'alpha', 'period'}}
    K = gp_kernel_matrix(coords, coords, kernel_type=kernel_type, **kernel_kwargs)

    return SpatialResult(
        name="GP-Kernel",
        statistic=float(np.mean(data)),
        extra={
            "kernel_type": kernel_type,
            "K_mean": float(np.mean(K)),
            "K_shape": tuple(K.shape),
            "n_points": n_points,
        },
    )


def cheatsheet() -> str:
    return "gpkrn(data=None, coords=None, n=50, kernel_type='se') -> SpatialResult with kernel matrix summary"
