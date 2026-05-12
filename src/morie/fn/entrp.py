# morie.fn -- function file (hadesllm/morie)
"""Metric entropy (covering number) computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MetricEntropyResult:
    """Result of metric entropy computation.

    Attributes
    ----------
    log_covering_number : float
        Log covering number at the given epsilon.
    epsilon : float
        Ball radius for covering.
    covering_number : int
        Estimated number of epsilon-balls needed.
    entropy_integral : float
        Numerical approximation of the entropy integral from epsilon to
        the diameter.
    diameter : float
        Estimated diameter of the set.
    dimension : int
        Ambient dimension.
    """

    log_covering_number: float
    epsilon: float
    covering_number: int
    entropy_integral: float
    diameter: float
    dimension: int


def entrp(
    X: np.ndarray,
    *,
    epsilon: float = 0.1,
    metric: str = "L2",
    n_grid: int = 50,
) -> MetricEntropyResult:
    r"""
    Estimate the metric entropy (log covering number) of a point set.

    The :math:`\varepsilon`-covering number
    :math:`N(\varepsilon, T, d)` is the minimum number of
    :math:`d`-balls of radius :math:`\varepsilon` needed to cover :math:`T`.
    The metric entropy is:

    .. math::

        H(\varepsilon, T, d) = \log N(\varepsilon, T, d)

    For a bounded subset of :math:`\mathbb{R}^d` with diameter :math:`D`,
    a volume argument gives:

    .. math::

        \log N(\varepsilon, T, \|\cdot\|_2)
        \le d \log\!\left(\frac{2D}{\varepsilon} + 1\right)

    The entropy integral controls uniform law of large numbers rates:

    .. math::

        J(\delta, \mathcal{F}, L_2)
        = \int_0^\delta \sqrt{\log N(\varepsilon, \mathcal{F}, L_2)}
          \, d\varepsilon

    :param X: Data matrix (n, d) or point set whose covering is estimated.
    :param epsilon: Ball radius. Default 0.1.
    :param metric: Distance metric. ``"L2"`` (Euclidean). Default ``"L2"``.
    :param n_grid: Grid points for entropy integral. Default 50.
    :return: MetricEntropyResult with log covering number and entropy integral.
    :raises ValueError: If X is empty or epsilon <= 0.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 9.1 (Covering numbers, metric entropy).
    Springer. DOI:10.1007/978-0-387-74978-5

    van der Vaart, A.W. & Wellner, J.A. (1996). *Weak Convergence and
    Empirical Processes*, Sec. 2.2. Springer.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.size == 0:
        raise ValueError("X must be non-empty.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if metric != "L2":
        raise ValueError(f"Only 'L2' metric is supported, got '{metric}'.")

    n, d = X.shape

    col_ranges = np.ptp(X, axis=0)
    diameter = float(np.linalg.norm(col_ranges))
    if diameter < 1e-15:
        diameter = 1.0

    log_cover = d * np.log(2.0 * diameter / epsilon + 1.0)
    covering_number = int(np.ceil(np.exp(log_cover)))

    eps_grid = np.linspace(epsilon, diameter, n_grid)
    log_covers_grid = d * np.log(2.0 * diameter / eps_grid + 1.0)
    log_covers_grid = np.maximum(log_covers_grid, 0.0)
    integrand = np.sqrt(log_covers_grid)
    entropy_integral = float(np.trapezoid(integrand, eps_grid))

    return MetricEntropyResult(
        log_covering_number=float(log_cover),
        epsilon=epsilon,
        covering_number=covering_number,
        entropy_integral=entropy_integral,
        diameter=diameter,
        dimension=d,
    )


def cheatsheet() -> str:
    return "entrp({X}) -> Metric entropy (covering number) computation."
