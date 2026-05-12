# morie.fn -- function file (hadesllm/morie)
"""Empirical bootstrap process.

Computes the bootstrap empirical process and tests for its weak
convergence to a Brownian bridge, which underpins the validity of
bootstrap-based inference for empirical process functionals.

References
----------
Gine, E. & Zinn, J. (1990). Bootstrapping general empirical measures.
*Annals of Probability*, 18(2), 851--869.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapters 2, 10.

van der Vaart, A. W. & Wellner, J. A. (1996). *Weak Convergence and
Empirical Processes*. Springer. Chapters 2.9, 3.6.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def empbs(
    data: np.ndarray,
    *,
    n_boot: int = 499,
    eval_points: np.ndarray | None = None,
    n_eval: int = 100,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    r"""Compute the empirical bootstrap process and confidence bands.

    The empirical process is :math:`\mathbb{G}_n(t) = \sqrt{n}(\hat{F}_n(t) - F(t))`.
    The bootstrap version is:

    .. math::

        \mathbb{G}_n^*(t) = \sqrt{n}(\hat{F}_n^*(t) - \hat{F}_n(t))

    Under regularity conditions (Gine & Zinn, 1990), the conditional
    distribution of :math:`\|\mathbb{G}_n^*\|_\infty` consistently
    estimates that of :math:`\|\mathbb{G}_n\|_\infty`, enabling
    construction of simultaneous confidence bands for the CDF.

    Parameters
    ----------
    data : np.ndarray
        Observed data, shape ``(n,)``.
    n_boot : int
        Number of bootstrap replicates.
    eval_points : np.ndarray | None
        Points at which to evaluate the empirical process.
        If *None*, uses an equispaced grid over the data range.
    n_eval : int
        Number of evaluation points if *eval_points* is not supplied.
    alpha : float
        Significance level for simultaneous confidence bands.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        ``eval_points``, ``ecdf`` (empirical CDF at eval points),
        ``cb_lower``, ``cb_upper`` (simultaneous confidence band),
        ``sup_stats`` (array of bootstrap sup statistics),
        ``ks_critical_value``, ``n``, ``n_boot``.
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    rng = np.random.default_rng(seed)

    if eval_points is None:
        lo, hi = float(np.min(data)), float(np.max(data))
        margin = 0.05 * (hi - lo) if hi > lo else 1.0
        eval_points = np.linspace(lo - margin, hi + margin, n_eval)
    else:
        eval_points = np.asarray(eval_points, dtype=float)

    ecdf_vals = np.array([float(np.mean(data <= t)) for t in eval_points])

    sup_stats = np.empty(n_boot)
    boot_processes = np.empty((n_boot, len(eval_points)))

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_sample = data[idx]
        boot_ecdf = np.array([float(np.mean(boot_sample <= t)) for t in eval_points])
        process = np.sqrt(n) * (boot_ecdf - ecdf_vals)
        boot_processes[b] = process
        sup_stats[b] = float(np.max(np.abs(process)))

    ks_crit = float(np.percentile(sup_stats, 100 * (1 - alpha)))

    cb_lower = ecdf_vals - ks_crit / np.sqrt(n)
    cb_upper = ecdf_vals + ks_crit / np.sqrt(n)
    cb_lower = np.clip(cb_lower, 0.0, 1.0)
    cb_upper = np.clip(cb_upper, 0.0, 1.0)

    return {
        "eval_points": eval_points,
        "ecdf": ecdf_vals,
        "cb_lower": cb_lower,
        "cb_upper": cb_upper,
        "sup_stats": sup_stats,
        "ks_critical_value": ks_crit,
        "n": n,
        "n_boot": n_boot,
    }


empbs_fn = empbs


def cheatsheet() -> str:
    return "empbs(data) -> Empirical bootstrap process + confidence bands (Kosorok 2008, Ch. 2, 10)."
