# morie.fn -- function file (hadesllm/morie)
"""Empirical CDF with Dvoretzky-Kiefer-Wolfowitz confidence bands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ECDFResult:
    """Result of empirical CDF computation.

    Attributes
    ----------
    x_sorted : np.ndarray
        Sorted unique data values.
    ecdf_vals : np.ndarray
        ECDF evaluated at each sorted value.
    lower : np.ndarray
        Lower DKW confidence band.
    upper : np.ndarray
        Upper DKW confidence band.
    alpha : float
        Significance level used for bands.
    n : int
        Sample size.
    epsilon : float
        DKW band half-width.
    """

    x_sorted: np.ndarray
    ecdf_vals: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    n: int
    epsilon: float


def ecdf(
    x: np.ndarray,
    *,
    alpha: float = 0.05,
) -> ECDFResult:
    r"""
    Compute the empirical cumulative distribution function with DKW confidence bands.

    The empirical CDF at point :math:`t` is:

    .. math::

        \hat{F}_n(t) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(X_i \le t)

    The Dvoretzky-Kiefer-Wolfowitz (DKW) inequality provides a simultaneous
    confidence band:

    .. math::

        P\!\left(\sup_t |\hat{F}_n(t) - F(t)| > \varepsilon\right)
        \le 2\exp(-2n\varepsilon^2)

    Setting the right-hand side to :math:`\alpha` yields
    :math:`\varepsilon = \sqrt{\ln(2/\alpha) / (2n)}`.

    :param x: 1-D array of observations.
    :param alpha: Significance level for the confidence band. Default 0.05.
    :return: ECDFResult with sorted values, ECDF, lower/upper bands, and metadata.
    :raises ValueError: If x is empty or alpha not in (0, 1).

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 2. Springer. DOI:10.1007/978-0-387-74978-5

    Dvoretzky, A., Kiefer, J., & Wolfowitz, J. (1956). Asymptotic minimax
    character of the sample distribution function and of the classical
    multinomial estimator. *Annals of Mathematical Statistics*, 27(3), 642--669.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    n = x.size
    x_sorted = np.sort(x)
    ecdf_vals = np.arange(1, n + 1) / n

    epsilon = np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
    lower = np.clip(ecdf_vals - epsilon, 0.0, 1.0)
    upper = np.clip(ecdf_vals + epsilon, 0.0, 1.0)

    return ECDFResult(
        x_sorted=x_sorted,
        ecdf_vals=ecdf_vals,
        lower=lower,
        upper=upper,
        alpha=alpha,
        n=n,
        epsilon=epsilon,
    )


def cheatsheet() -> str:
    return "ecdf({x}) -> Empirical CDF with DKW confidence bands."
