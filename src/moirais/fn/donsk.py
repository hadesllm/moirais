# moirais.fn — function file (hadesllm/moirais)
"""Donsker class membership test via bootstrap empirical process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DonskerResult:
    """Result of Donsker class membership assessment.

    Attributes
    ----------
    observed_sup : float
        Supremum norm of the observed empirical process.
    bootstrap_quantile : float
        Bootstrap critical value at the (1 - alpha) quantile.
    p_value : float
        Fraction of bootstrap suprema exceeding the observed supremum.
    is_donsker : bool
        True if observed_sup <= bootstrap_quantile (no evidence against
        Donsker membership).
    n : int
        Sample size.
    n_boot : int
        Number of bootstrap replications.
    """

    observed_sup: float
    bootstrap_quantile: float
    p_value: float
    is_donsker: bool
    n: int
    n_boot: int


def donsk(x: np.ndarray, cdf=None, *, cdf_fn: object | None = None, n_boot: int = 500, alpha: float = 0.05, seed: int | None = None) -> DonskerResult:
    r"""
    Test Donsker class membership via multiplier bootstrap of the empirical process.

    A class of functions :math:`\mathcal{F}` is Donsker if the empirical
    process :math:`\mathbb{G}_n f = \sqrt{n}(P_n - P)f` converges weakly
    to a tight Gaussian process :math:`\mathbb{G}` in
    :math:`\ell^\infty(\mathcal{F})`.

    This function tests the specific case :math:`\mathcal{F} = \{1_{(-\infty,t]}:
    t \in \mathbb{R}\}` (indicators), which is always Donsker. For finite
    samples, the bootstrap calibration detects whether the empirical process
    behaves like a Brownian bridge. A large observed supremum relative to the
    bootstrap distribution suggests non-regularity.

    The multiplier bootstrap generates:

    .. math::

        \mathbb{G}_n^*(t) = \frac{1}{\sqrt{n}} \sum_{i=1}^n
        (w_i - 1)\bigl(\mathbf{1}(X_i \le t) - \hat{F}_n(t)\bigr)

    where :math:`w_i \sim \text{Exp}(1)` (Bayesian bootstrap weights).

    :param x: 1-D array of observations.
    :param cdf_fn: Reference CDF callable. If None, standard normal is used.
    :param n_boot: Number of bootstrap replications. Default 500.
    :param alpha: Significance level. Default 0.05.
    :param seed: Random seed for reproducibility.
    :return: DonskerResult with observed supremum, bootstrap quantile, p-value.
    :raises ValueError: If x is empty or n_boot < 1.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 2.5 (Donsker classes). Springer.
    DOI:10.1007/978-0-387-74978-5

    van der Vaart, A.W. & Wellner, J.A. (1996). *Weak Convergence and
    Empirical Processes*, Sec. 2.5. Springer.
    """
    from scipy.stats import norm as _norm

    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}.")

    rng = np.random.default_rng(seed)
    n = x.size
    if cdf_fn is None:
        cdf_fn = _norm.cdf

    x_sorted = np.sort(x)
    ecdf_vals = np.arange(1, n + 1) / n
    true_vals = np.asarray(cdf_fn(x_sorted))
    observed_process = np.sqrt(n) * (ecdf_vals - true_vals)
    observed_sup = float(np.max(np.abs(observed_process)))

    indicators = (x[:, None] <= x_sorted[None, :]).astype(float)

    boot_sups = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.exponential(1.0, size=n)
        centered = indicators - ecdf_vals[None, :]
        gn_star = (w - 1.0) @ centered / np.sqrt(n)
        boot_sups[b] = np.max(np.abs(gn_star))

    bootstrap_q = float(np.quantile(boot_sups, 1.0 - alpha))
    p_value = float(np.mean(boot_sups >= observed_sup))

    return DonskerResult(
        observed_sup=observed_sup,
        bootstrap_quantile=bootstrap_q,
        p_value=p_value,
        is_donsker=observed_sup <= bootstrap_q,
        n=n,
        n_boot=n_boot,
    )


def cheatsheet() -> str:
    return "donsk({x}) -> Donsker class membership test via bootstrap."
