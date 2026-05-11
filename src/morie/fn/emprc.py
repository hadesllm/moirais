# morie.fn — function file (hadesllm/morie)
"""Empirical process: centered and scaled ECDF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EmpiricalProcessResult:
    """Result of empirical process computation.

    Attributes
    ----------
    t_grid : np.ndarray
        Evaluation grid points.
    process : np.ndarray
        Centered and scaled empirical process values at each grid point.
    sup_norm : float
        Supremum norm of the process over the grid.
    n : int
        Sample size.
    """

    t_grid: np.ndarray
    process: np.ndarray
    sup_norm: float
    n: int


def emprc(x: np.ndarray, cdf=None, *, cdf_fn: object | None = None, n_grid: int = 200) -> EmpiricalProcessResult:
    r"""
    Compute the centered, scaled empirical process.

    The empirical process is defined as:

    .. math::

        \mathbb{G}_n(t) = \sqrt{n}\bigl(\hat{F}_n(t) - F(t)\bigr)

    where :math:`\hat{F}_n` is the empirical CDF and :math:`F` is the true CDF.
    Under regularity conditions, :math:`\mathbb{G}_n` converges weakly to a
    Brownian bridge (Donsker's theorem).

    When ``cdf_fn`` is None, the standard normal CDF is used as the reference
    distribution (useful for goodness-of-fit assessment against normality).

    :param x: 1-D array of observations.
    :param cdf_fn: Callable ``cdf_fn(t) -> float/array`` for the reference CDF.
        If None, the standard normal CDF is used.
    :param n_grid: Number of grid points for evaluation. Default 200.
    :return: EmpiricalProcessResult with grid, process values, and supremum norm.
    :raises ValueError: If x is empty or n_grid < 2.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 2, Def. 2.1. Springer.
    DOI:10.1007/978-0-387-74978-5

    van der Vaart, A.W. (1998). *Asymptotic Statistics*, Ch. 19.
    Cambridge University Press.
    """
    from scipy.stats import norm as _norm

    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}.")

    n = x.size
    if cdf_fn is None:
        cdf_fn = _norm.cdf

    t_grid = np.linspace(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5, n_grid)

    ecdf_vals = np.array([np.mean(x <= t) for t in t_grid])
    true_vals = np.array([cdf_fn(t) for t in t_grid])

    process = np.sqrt(n) * (ecdf_vals - true_vals)
    sup_norm = float(np.max(np.abs(process)))

    return EmpiricalProcessResult(
        t_grid=t_grid,
        process=process,
        sup_norm=sup_norm,
        n=n,
    )


def cheatsheet() -> str:
    return "emprc({x}) -> Centered, scaled empirical process."
