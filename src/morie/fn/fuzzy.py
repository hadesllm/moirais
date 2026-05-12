# morie.fn -- function file (hadesllm/morie)
"""Fuzzy membership function evaluation."""

import numpy as np

from ._containers import DescriptiveResult
def fuzzy_membership(
    x,
    mf_type: str = "triangular",
    params=None,
    **kwargs,
) -> DescriptiveResult:
    """
    Evaluate a fuzzy membership function at given points.

    Supported types:

    - **triangular**: params = (a, b, c) where a <= b <= c
    - **trapezoidal**: params = (a, b, c, d) where a <= b <= c <= d
    - **gaussian**: params = (mean, sigma)
    - **sigmoid**: params = (a, c) where a = slope, c = crossover

    :param x: Input values at which to evaluate membership.
    :param mf_type: Type of membership function. Default ``"triangular"``.
    :param params: Parameters for the chosen function type.
    :return: DescriptiveResult with membership degrees.
    :raises ValueError: If mf_type is unknown or params are invalid.

    References
    ----------
    Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.
    Mendel, J. M. (2017). *Uncertain Rule-Based Fuzzy Systems* (2nd ed.).
    Springer.
    """
    x = np.asarray(x, dtype=np.float64)

    if mf_type == "triangular":
        if params is None:
            params = (0.0, 0.5, 1.0)
        a, b, c = params
        if not (a <= b <= c):
            raise ValueError(f"Need a <= b <= c, got {a}, {b}, {c}.")
        mu = np.zeros_like(x)
        left = b > a
        right = c > b
        if left:
            mask = (x >= a) & (x <= b)
            mu[mask] = (x[mask] - a) / (b - a)
        if right:
            mask = (x >= b) & (x <= c)
            mu[mask] = (c - x[mask]) / (c - b)
        mu[x == b] = 1.0

    elif mf_type == "trapezoidal":
        if params is None:
            params = (0.0, 0.25, 0.75, 1.0)
        a, b, c, d = params
        if not (a <= b <= c <= d):
            raise ValueError(f"Need a <= b <= c <= d, got {a}, {b}, {c}, {d}.")
        mu = np.zeros_like(x)
        if b > a:
            mask = (x >= a) & (x < b)
            mu[mask] = (x[mask] - a) / (b - a)
        mask = (x >= b) & (x <= c)
        mu[mask] = 1.0
        if d > c:
            mask = (x > c) & (x <= d)
            mu[mask] = (d - x[mask]) / (d - c)

    elif mf_type == "gaussian":
        if params is None:
            params = (0.5, 0.15)
        mean, sigma = params
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")
        mu = np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    elif mf_type == "sigmoid":
        if params is None:
            params = (10.0, 0.5)
        a, c = params
        mu = 1.0 / (1.0 + np.exp(-a * (x - c)))

    else:
        raise ValueError(f"Unknown mf_type '{mf_type}'. Use 'triangular', 'trapezoidal', 'gaussian', or 'sigmoid'.")

    return DescriptiveResult(
        name="fuzzy_membership",
        value=float(np.mean(mu)),
        extra={
            "membership": mu,
            "mf_type": mf_type,
            "params": params,
            "max_membership": float(np.max(mu)),
            "support_fraction": float(np.mean(mu > 0)),
        },
    )


fuzzy = fuzzy_membership


def cheatsheet() -> str:
    return "fuzzy_membership({}) -> Fuzzy membership function evaluation."
