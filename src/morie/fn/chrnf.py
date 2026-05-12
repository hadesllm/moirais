# morie.fn -- function file (hadesllm/morie)
"""Chernoff distribution (cube-root asymptotics)."""

from __future__ import annotations

import numpy as np


def chernoff_distribution(
    x: np.ndarray,
    *,
    n_grid: int = 200,
    x_range: tuple[float, float] = (-4.0, 4.0),
) -> dict:
    r"""
    Approximate the Chernoff distribution (cube-root asymptotics).

    The Chernoff distribution arises as the limit distribution of
    change-point estimators and monotone density estimators at rate
    :math:`n^{1/3}`. It is the distribution of:

    .. math::

        Z = \arg\max_{t \in \mathbb{R}} \{W(t) - t^2\}

    where :math:`W(t)` is standard two-sided Brownian motion.

    This function estimates the Chernoff distribution via Monte Carlo
    simulation of discretized Brownian motion.

    :param x: Evaluation points for the approximate CDF/PDF.
    :param n_grid: Grid resolution for Brownian motion simulation.
        Default 200.
    :param x_range: Range of Brownian motion grid. Default (-4, 4).
    :return: dict with ``x`` (evaluation points), ``cdf`` (approximate CDF),
        ``pdf`` (approximate PDF via finite differences),
        ``mean`` (approximate mean), ``variance``.
    :raises ValueError: If x is empty.

    References
    ----------
    Chernoff, H. (1964). Estimation of the mode. *Ann. Inst. Statist.
        Math.*, 16, 31--41.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 14.4 (cube-root asymptotics).
        Springer.
    Groeneboom, P. & Wellner, J.A. (2001). Computing Chernoff's
        distribution. *J. Comput. Graph. Statist.*, 10(2), 388--400.
    """
    x_eval = np.asarray(x, dtype=float).ravel()
    if x_eval.size == 0:
        raise ValueError("x must be non-empty.")

    grid = np.linspace(x_range[0], x_range[1], n_grid)
    dt = grid[1] - grid[0]

    n_mc = 5000
    rng = np.random.default_rng(42)
    argmaxes = np.empty(n_mc)

    for k in range(n_mc):
        increments = rng.standard_normal(n_grid - 1) * np.sqrt(dt)
        W = np.zeros(n_grid)
        W[1:] = np.cumsum(increments)
        drift = W - grid**2
        argmaxes[k] = grid[np.argmax(drift)]

    cdf = np.array([float(np.mean(argmaxes <= xi)) for xi in x_eval])

    dx = np.diff(x_eval)
    dx[dx == 0] = 1e-10
    pdf_interior = np.diff(cdf) / dx
    pdf = np.zeros_like(cdf)
    pdf[:-1] = pdf_interior
    pdf[-1] = pdf_interior[-1] if len(pdf_interior) > 0 else 0.0
    pdf = np.maximum(pdf, 0.0)

    return {
        "x": x_eval.tolist(),
        "cdf": cdf.tolist(),
        "pdf": pdf.tolist(),
        "mean": float(np.mean(argmaxes)),
        "variance": float(np.var(argmaxes, ddof=1)),
        "n_mc": n_mc,
        "method": "Chernoff distribution (Monte Carlo)",
    }


chrnf = chernoff_distribution


def cheatsheet() -> str:
    return "chernoff_distribution({x}) -> Chernoff distribution (cube-root asymptotics)."
