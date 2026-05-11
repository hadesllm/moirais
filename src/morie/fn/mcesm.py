# morie.fn — function file (hadesllm/morie)
"""Maximum Calibration Error (MCE) for probability calibration.

MCE is the worst-case calibration error across all probability bins,
measuring the maximum discrepancy between confidence and accuracy.

References
----------
Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining
well calibrated probabilities using Bayesian binning.
*Proceedings of AAAI*, 29, 2901-2907.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
calibration of modern neural networks.
*Proceedings of ICML*, 70, 1321-1330.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["mcesm"]


def mcesm(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    min_bin_count: int = 5,
) -> dict[str, Any]:
    r"""Compute the Maximum Calibration Error (MCE).

    .. math::

        \mathrm{MCE} = \max_{b \in \{1,\ldots,B\}}
            |\mathrm{acc}(b) - \mathrm{conf}(b)|

    Only bins with at least ``min_bin_count`` observations are
    considered to avoid noise from sparse bins.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels ``{0, 1}``, shape ``(n,)``.
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n,)``.
    n_bins : int
        Number of probability bins.
    strategy : str
        ``"uniform"`` (equal-width) or ``"quantile"`` (equal-freq).
    min_bin_count : int
        Minimum observations per bin to include in MCE.

    Returns
    -------
    dict
        ``mce``, ``worst_bin_idx``, ``worst_bin_center``,
        ``worst_acc``, ``worst_conf``,
        ``bin_gaps`` (calibration gap per bin),
        ``bin_counts``, ``n``, ``method``.

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

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        n_bins = len(edges) - 1
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    bin_centers = np.empty(n_bins)
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    gaps = np.full(n_bins, np.nan)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        bin_centers[i] = (lo + hi) / 2.0
        counts[i] = int(mask.sum())
        if counts[i] >= min_bin_count:
            accs[i] = float(y_true[mask].mean())
            confs[i] = float(y_prob[mask].mean())
            gaps[i] = abs(accs[i] - confs[i])

    valid = ~np.isnan(gaps)
    if not valid.any():
        mce = float("nan")
        worst_idx = -1
    else:
        worst_idx = int(np.nanargmax(gaps))
        mce = float(gaps[worst_idx])

    return {
        "mce": mce,
        "worst_bin_idx": worst_idx,
        "worst_bin_center": float(bin_centers[worst_idx]) if worst_idx >= 0 else float("nan"),
        "worst_acc": float(accs[worst_idx]) if worst_idx >= 0 else float("nan"),
        "worst_conf": float(confs[worst_idx]) if worst_idx >= 0 else float("nan"),
        "bin_gaps": gaps,
        "bin_counts": counts,
        "n": n,
        "method": f"MCE-{strategy}",
    }


def cheatsheet() -> str:
    return "mcesm(y_true, y_prob) -> Maximum calibration error (Naeini et al. 2015, AAAI)."
