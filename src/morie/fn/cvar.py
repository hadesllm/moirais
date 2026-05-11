# morie.fn — function file (hadesllm/morie)
"""Conditional VaR (Expected Shortfall)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def conditional_var(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> DescriptiveResult:
    """Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR:

    .. math::

        \\text{CVaR}_\\alpha = -\\frac{1}{\\alpha}
        \\int_0^\\alpha F^{-1}(p) \\, dp
        = -E[R \\mid R \\leq -\\text{VaR}_\\alpha]

    Parameters
    ----------
    returns : array-like
        Array of portfolio returns.
    alpha : float, default 0.05
        Significance level (e.g. 0.05 for 95% CVaR).

    Returns
    -------
    DescriptiveResult
        ``value`` is the CVaR (positive = loss magnitude).
        ``extra`` has ``var`` (the VaR threshold), ``alpha``,
        ``n_tail`` (number of tail observations).

    Raises
    ------
    ValueError
        If alpha out of (0, 1) or too few observations.

    References
    ----------
    Acerbi, C., & Tasche, D. (2002). On the coherence of expected
    shortfall. *Journal of Banking & Finance*, 26(7), 1487--1503.

    Rockafellar, R. T., & Uryasev, S. (2000). Optimization of
    conditional value-at-risk. *Journal of Risk*, 2(3), 21--42.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    if len(r) < 2:
        raise ValueError("Need at least 2 return observations.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    cutoff = np.percentile(r, 100 * alpha)
    tail = r[r <= cutoff]
    n_tail = len(tail)

    if n_tail == 0:
        cvar = -float(cutoff)
    else:
        cvar = -float(np.mean(tail))

    var = -float(cutoff)

    return DescriptiveResult(
        name="ConditionalVaR",
        value=float(cvar),
        extra={
            "var": float(var),
            "alpha": alpha,
            "n_tail": n_tail,
            "n_obs": len(r),
        },
    )


cvar = conditional_var


def cheatsheet() -> str:
    return "conditional_var({}) -> Conditional VaR (Expected Shortfall)."
