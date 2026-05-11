# morie.fn — function file (hadesllm/morie)
"""Geweke convergence diagnostic."""

from __future__ import annotations

__all__ = ["geweke_diagnostic", "gewek"]

from typing import Any, Union

import numpy as np
from scipy import stats


def geweke_diagnostic(samples: Union[list, np.ndarray], cdf=None, *, first_frac: float = 0.1, last_frac: float = 0.5) -> dict[str, Any]:
    """
    Geweke (1992) convergence diagnostic.

    Compares means of the first and last portions of a single chain
    using a two-sample z-test with spectral density variance
    estimates.  If the chain has converged, the two means should
    be indistinguishable.

    Parameters
    ----------
    samples : array-like
        MCMC samples (1-D).
    first_frac : float
        Fraction of chain for the early window (default 0.1).
    last_frac : float
        Fraction of chain for the late window (default 0.5).

    Returns
    -------
    dict
        z_score, p_value, converged (|z| < 1.96), mean_first,
        mean_last

    Raises
    ------
    ValueError
        If fractions sum to > 1 or chain too short.

    References
    ----------
    Geweke, J. (1992). Evaluating the accuracy of sampling-based
    approaches to the calculation of posterior moments.
    *Bayesian Statistics 4*, Oxford University Press, 169--193.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    if n < 10:
        raise ValueError("Chain must have at least 10 samples.")
    if first_frac + last_frac > 1.0:
        raise ValueError("first_frac + last_frac must be <= 1.")

    n_a = max(int(first_frac * n), 1)
    n_b = max(int(last_frac * n), 1)

    a = x[:n_a]
    b = x[-n_b:]

    def _spectral_var(arr):
        """Variance of the mean using batch means (Geyer, 1992)."""
        m = len(arr)
        if m < 2:
            return float(np.var(arr, ddof=0)) + 1e-30
        batch_size = max(int(np.sqrt(m)), 2)
        n_batches = m // batch_size
        if n_batches < 2:
            return float(np.var(arr, ddof=1) / m) + 1e-30
        batch_means = np.array([
            np.mean(arr[i * batch_size:(i + 1) * batch_size])
            for i in range(n_batches)
        ])
        return float(batch_size * np.var(batch_means, ddof=1) / m) + 1e-30

    var_a = _spectral_var(a)
    var_b = _spectral_var(b)

    z = float((np.mean(a) - np.mean(b)) / np.sqrt(var_a + var_b))
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "z_score": z,
        "p_value": p,
        "converged": bool(abs(z) < 1.96),
        "mean_first": float(np.mean(a)),
        "mean_last": float(np.mean(b)),
    }


gewek = geweke_diagnostic


def cheatsheet() -> str:
    return "geweke_diagnostic(samples) -> Geweke convergence diagnostic."
