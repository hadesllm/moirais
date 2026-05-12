# morie.fn -- function file (hadesllm/morie)
"""Effective sample size for MCMC chains."""

from __future__ import annotations

__all__ = ["effective_sample_size", "ess"]

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def effective_sample_size(
    samples: Union[list, np.ndarray],
) -> dict[str, Any]:
    r"""
    Compute the effective sample size (ESS) for an MCMC chain.

    ESS quantifies the number of independent samples equivalent
    to the correlated chain.  Uses the initial positive sequence
    estimator (IPSE) of Geyer (1992):

    .. math::

        \\text{ESS} = \\frac{n}{1 + 2 \\sum_{k=1}^{K} \\hat{\\rho}(k)}

    where the sum is truncated at the first negative pair of
    autocorrelations.

    Parameters
    ----------
    samples : array-like
        MCMC samples (1-D array of length n).

    Returns
    -------
    dict
        ess : float -- effective sample size
        n : int -- total chain length
        efficiency : float -- ESS / n

    Raises
    ------
    ValueError
        If samples has fewer than 4 elements.

    References
    ----------
    Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
    *Statistical Science*, 7(4), 473--483.
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    Ch. 11.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    if n < 4:
        raise ValueError("Need at least 4 samples.")

    x_centered = x - np.mean(x)
    var_x = np.var(x, ddof=0)
    if var_x < 1e-30:
        return RichResult(payload={"ess": float(n), "n": n, "efficiency": 1.0})

    fft_x = np.fft.rfft(x_centered, n=2 * n)
    acf_full = np.fft.irfft(np.abs(fft_x) ** 2)[:n]
    acf_full /= acf_full[0]

    sum_rho = 0.0
    k = 1
    while k < n - 1:
        pair_sum = acf_full[k] + (acf_full[k + 1] if k + 1 < n else 0.0)
        if pair_sum < 0:
            break
        sum_rho += pair_sum
        k += 2

    tau = 1.0 + 2.0 * sum_rho
    ess_val = max(1.0, n / tau)

    return {
        "ess": float(ess_val),
        "n": n,
        "efficiency": float(ess_val / n),
    }


ess = effective_sample_size


def cheatsheet() -> str:
    return "effective_sample_size(samples) -> MCMC effective sample size (ESS)."
