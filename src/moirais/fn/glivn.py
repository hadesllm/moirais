# moirais.fn — function file (hadesllm/moirais)
"""Glivenko-Cantelli uniform convergence test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GlivenkoCantelliResult:
    """Result of Glivenko-Cantelli convergence assessment.

    Attributes
    ----------
    sup_deviation : float
        Supremum deviation between ECDF and reference CDF.
    dkw_bound : float
        DKW probabilistic upper bound at significance alpha.
    converges : bool
        True if sup_deviation <= dkw_bound (consistent with GC theorem).
    n : int
        Sample size.
    alpha : float
        Significance level used.
    """

    sup_deviation: float
    dkw_bound: float
    converges: bool
    n: int
    alpha: float


def glivn(x: np.ndarray, cdf=None, *, cdf_fn: object | None = None, alpha: float = 0.05) -> GlivenkoCantelliResult:
    r"""
    Assess Glivenko-Cantelli uniform convergence of the ECDF to a reference CDF.

    The Glivenko-Cantelli theorem states that for i.i.d. data:

    .. math::

        \|\hat{F}_n - F\|_\infty
        = \sup_t |\hat{F}_n(t) - F(t)| \xrightarrow{a.s.} 0

    This function computes the supremum deviation and compares it against
    the Dvoretzky-Kiefer-Wolfowitz finite-sample bound:

    .. math::

        \varepsilon = \sqrt{\frac{\ln(2/\alpha)}{2n}}

    If the observed supremum is below the bound, the data are consistent
    with the reference CDF at level :math:`\alpha`.

    :param x: 1-D array of observations.
    :param cdf_fn: Callable ``cdf_fn(t) -> float/array``. If None, standard
        normal CDF is used.
    :param alpha: Significance level. Default 0.05.
    :return: GlivenkoCantelliResult with deviation, bound, and convergence flag.
    :raises ValueError: If x is empty or alpha not in (0, 1).

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Theorem 2.4 (Glivenko-Cantelli). Springer.
    DOI:10.1007/978-0-387-74978-5

    Glivenko, V. (1933). Sulla determinazione empirica delle leggi di
    probabilita. *Giornale dell'Istituto Italiano degli Attuari*, 4, 92--99.

    Cantelli, F.P. (1933). Sulla determinazione empirica delle leggi di
    probabilita. *Giornale dell'Istituto Italiano degli Attuari*, 4, 421--424.
    """
    from scipy.stats import norm as _norm

    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    n = x.size
    if cdf_fn is None:
        cdf_fn = _norm.cdf

    x_sorted = np.sort(x)
    ecdf_vals = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n

    true_vals = np.asarray(cdf_fn(x_sorted))

    sup_dev = float(max(
        np.max(np.abs(ecdf_vals - true_vals)),
        np.max(np.abs(ecdf_prev - true_vals)),
    ))

    dkw_bound = np.sqrt(np.log(2.0 / alpha) / (2.0 * n))

    return GlivenkoCantelliResult(
        sup_deviation=sup_dev,
        dkw_bound=float(dkw_bound),
        converges=bool(sup_dev <= dkw_bound),
        n=n,
        alpha=alpha,
    )


def cheatsheet() -> str:
    return "glivn({x}) -> Glivenko-Cantelli uniform convergence test."
