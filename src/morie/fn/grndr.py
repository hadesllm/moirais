# morie.fn — function file (hadesllm/morie)
"""Grenander estimator (NPMLE of monotone decreasing density)."""

from __future__ import annotations

import numpy as np


def grenander_estimator(
    x: np.ndarray,
) -> dict:
    r"""
    Grenander estimator: NPMLE of a monotone non-increasing density.

    Given iid observations from a density :math:`f` that is non-increasing
    on :math:`[0, \infty)`, the Grenander estimator is the left derivative
    of the least concave majorant (LCM) of the empirical CDF:

    .. math::

        \hat{f}_n(t) = \frac{d}{dt}\, \text{LCM}(\hat{F}_n)(t)

    The estimator converges at the cube-root rate :math:`n^{1/3}` to
    a Chernoff-type distribution at points where :math:`f'(t) < 0`.

    :param x: 1-D array of non-negative observations.
    :return: dict with ``knots`` (change points of the LCM),
        ``heights`` (density values on each interval),
        ``lcm_x``, ``lcm_y`` (LCM of the ECDF).
    :raises ValueError: If x is empty or contains negative values.

    References
    ----------
    Grenander, U. (1956). On the theory of mortality measurement, Part II.
        *Skand. Aktuarietidskr.*, 39, 125--153.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 14.4. Springer.
    Groeneboom, P. & Jongbloed, G. (2014). *Nonparametric Estimation
        under Shape Constraints*. Cambridge Univ. Press.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if np.any(x < 0):
        raise ValueError("Grenander estimator requires non-negative data.")

    n = x.size
    x_sorted = np.sort(x)
    ecdf_x = np.concatenate([[0.0], x_sorted])
    ecdf_y = np.concatenate([[0.0], np.arange(1, n + 1) / n])

    lcm_x, lcm_y = _least_concave_majorant(ecdf_x, ecdf_y)

    knots = []
    heights = []
    for i in range(len(lcm_x) - 1):
        dx = lcm_x[i + 1] - lcm_x[i]
        if dx > 0:
            slope = (lcm_y[i + 1] - lcm_y[i]) / dx
            knots.append(float(lcm_x[i]))
            heights.append(float(slope))

    return {
        "knots": knots,
        "heights": heights,
        "lcm_x": lcm_x.tolist(),
        "lcm_y": lcm_y.tolist(),
        "n": n,
        "method": "Grenander estimator (LCM of ECDF)",
    }


def _least_concave_majorant(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the least concave majorant of (x, y) via upper convex hull."""
    n = len(x)
    hull = [0]
    for i in range(1, n):
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            dx1 = x[k] - x[j]
            dy1 = y[k] - y[j]
            dx2 = x[i] - x[j]
            dy2 = y[i] - y[j]
            if dx1 * dy2 >= dx2 * dy1:
                hull.pop()
            else:
                break
        hull.append(i)

    return x[hull], y[hull]


grndr = grenander_estimator


def cheatsheet() -> str:
    return "grenander_estimator({x}) -> Grenander estimator (monotone density NPMLE)."
