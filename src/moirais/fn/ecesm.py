# moirais.fn — function file (hadesllm/moirais)
"""Expected Calibration Error (ECE) and variants.

ECE measures the weighted average absolute difference between
predicted confidence and empirical accuracy across probability bins.

References
----------
Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining
well calibrated probabilities using Bayesian binning.
*Proceedings of AAAI*, 29, 2901-2907.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
calibration of modern neural networks.
*Proceedings of ICML*, 70, 1321-1330.

Kumar, A., Liang, P., & Ma, T. (2019). Verified uncertainty
calibration. *Advances in NeurIPS*, 32.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ecesm"]


def ecesm(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    norm: int = 1,
    strategy: str = "uniform",
    ci_method: str = "bootstrap",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Compute Expected Calibration Error (:math:`\ell_p`-ECE).

    .. math::

        \mathrm{ECE}_p = \left(\sum_{b=1}^{B}
            \frac{|B_b|}{n}
            |\text{acc}(b) - \text{conf}(b)|^p
        \right)^{1/p}

    For ``norm=1`` this is the standard ECE; for ``norm=2`` it is the
    root-mean-squared calibration error (RMSCE).

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels ``{0, 1}``, shape ``(n,)``.
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n,)``.
    n_bins : int
        Number of probability bins.
    norm : int
        :math:`\ell_p` norm (1 = ECE, 2 = RMSCE).
    strategy : str
        ``"uniform"`` (equal-width) or ``"quantile"`` (equal-frequency).
    ci_method : str
        ``"bootstrap"`` for bootstrap CI, ``"none"`` to skip.
    n_boot : int
        Bootstrap replicates.
    seed : int
        Random seed.
    alpha : float
        Significance level for CI.

    Returns
    -------
    dict
        ``ece``, ``ci_lower``, ``ci_upper`` (if bootstrap),
        ``bin_accs``, ``bin_confs``, ``bin_counts``,
        ``n``, ``n_bins``, ``norm``, ``method``.

    References
    ----------
    Naeini et al. (2015). AAAI 29, 2901-2907.
    Guo et al. (2017). ICML 70, 1321-1330.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0 and 1.")

    n = len(y_true)
    ece_val, accs, confs, counts = _compute_ece(y_true, y_prob, n_bins, norm, strategy)

    ci_lower, ci_upper = float("nan"), float("nan")
    if ci_method == "bootstrap":
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[b] = _compute_ece(y_true[idx], y_prob[idx], n_bins, norm, strategy)[0]
        z = float(np.abs(np.percentile(boot, 100 * alpha / 2)))
        ci_lower = max(0.0, ece_val - np.std(boot, ddof=1) * z)
        ci_upper = ece_val + np.std(boot, ddof=1) * float(
            np.abs(np.percentile((boot - boot.mean()) / boot.std(ddof=1), 100 * (1 - alpha / 2)))
        )
        # Simpler percentile CI
        ci_lower = float(np.percentile(boot, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    return {
        "ece": float(ece_val),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bin_accs": accs,
        "bin_confs": confs,
        "bin_counts": counts,
        "n": n,
        "n_bins": n_bins,
        "norm": norm,
        "method": f"ECE-L{norm}-{strategy}",
    }


def _compute_ece(y_true, y_prob, n_bins, norm, strategy):
    n = len(y_true)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        n_bins = len(edges) - 1

    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        counts[i] = mask.sum()
        if counts[i] > 0:
            accs[i] = float(y_true[mask].mean())
            confs[i] = float(y_prob[mask].mean())

    gap = np.abs(accs - confs)
    weights = counts / n
    if norm == 1:
        ece = float(np.sum(weights * gap))
    else:
        ece = float(np.sqrt(np.sum(weights * gap**norm)))
    return ece, accs, confs, counts


def cheatsheet() -> str:
    return "ecesm(y_true, y_prob) -> Expected calibration error (Naeini et al. 2015, AAAI)."
