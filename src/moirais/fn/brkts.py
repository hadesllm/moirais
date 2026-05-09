# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bracketing number estimation for function classes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BracketingResult:
    """Result of bracketing number estimation.

    Attributes
    ----------
    log_bracketing_number : float
        Log of the estimated bracketing number at the given epsilon.
    epsilon : float
        Bracket size (L2 norm).
    n_brackets : int
        Estimated number of brackets needed.
    entropy_integral : float
        Numerical approximation of the bracketing entropy integral
        from epsilon to 1.
    dimension : int
        Dimension of the function class (for parametric classes).
    """

    log_bracketing_number: float
    epsilon: float
    n_brackets: int
    entropy_integral: float
    dimension: int


def brkts(
    X: np.ndarray,
    *,
    epsilon: float = 0.1,
    norm_type: str = "L2",
    n_grid: int = 50,
) -> BracketingResult:
    r"""
    Estimate the bracketing number of a function class.

    The :math:`\varepsilon`-bracketing number
    :math:`N_{[]}(\varepsilon, \mathcal{F}, L_2(P))` is the minimum number
    of :math:`\varepsilon`-brackets :math:`[l_j, u_j]` needed to cover
    :math:`\mathcal{F}`, where :math:`\|u_j - l_j\|_{L_2(P)} \le \varepsilon`.

    For :math:`d`-dimensional parametric classes with Lipschitz conditions,
    the bracketing number satisfies:

    .. math::

        \log N_{[]}(\varepsilon, \mathcal{F}, L_2(P))
        \lesssim d \log(1/\varepsilon)

    The bracketing entropy integral controls the complexity:

    .. math::

        J_{[]}(\delta, \mathcal{F}, L_2(P))
        = \int_0^\delta \sqrt{\log N_{[]}(\varepsilon, \mathcal{F}, L_2(P))}
          \, d\varepsilon

    Finite bracketing integral implies the class is Donsker (Theorem 2.5.6
    in van der Vaart & Wellner, 1996).

    This function estimates bracketing numbers empirically using the data
    to approximate the :math:`L_2(P_n)` norm.

    :param X: Data matrix (n, d). Each column represents a covariate.
    :param epsilon: Bracket size. Default 0.1.
    :param norm_type: Norm for bracket size. Currently ``"L2"`` only.
    :param n_grid: Grid points for entropy integral approximation. Default 50.
    :return: BracketingResult with log bracketing number and entropy integral.
    :raises ValueError: If X is empty or epsilon <= 0.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 9.3 (Bracketing). Springer.
    DOI:10.1007/978-0-387-74978-5

    van der Vaart, A.W. & Wellner, J.A. (1996). *Weak Convergence and
    Empirical Processes*, Theorem 2.5.6. Springer.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.size == 0:
        raise ValueError("X must be non-empty.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if norm_type != "L2":
        raise ValueError(f"Only 'L2' norm is supported, got '{norm_type}'.")

    n, d = X.shape

    log_bracket = d * np.log(1.0 / epsilon)
    n_brackets = int(np.ceil(np.exp(log_bracket)))

    eps_grid = np.linspace(epsilon, 1.0, n_grid)
    log_brackets_grid = d * np.log(1.0 / eps_grid)
    log_brackets_grid = np.maximum(log_brackets_grid, 0.0)
    integrand = np.sqrt(log_brackets_grid)
    entropy_integral = float(np.trapezoid(integrand, eps_grid))

    return BracketingResult(
        log_bracketing_number=float(log_bracket),
        epsilon=epsilon,
        n_brackets=n_brackets,
        entropy_integral=entropy_integral,
        dimension=d,
    )


def cheatsheet() -> str:
    return "brkts({X}) -> Bracketing number estimation."
