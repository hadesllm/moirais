# moirais.fn — function file (hadesllm/moirais)
"""Reliability diagram values for probability calibration assessment.

A reliability diagram plots the fraction of positive outcomes against
the mean predicted probability within equal-width or equal-frequency
bins, diagnosing over- or under-confidence.

References
----------
DeGroot, M. H., & Fienberg, S. E. (1983). The comparison and
evaluation of forecasters. *Journal of the Royal Statistical Society,
Series D*, 32(1-2), 12-22.

Murphy, A. H., & Winkler, R. L. (1977). Reliability of subjective
probability forecasts of precipitation and temperature.
*Journal of the Royal Statistical Society, Series C*, 26(1), 41-47.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good
probabilities with supervised learning. *Proceedings of ICML*, 625-632.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["reldi"]


def reldi(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    r"""Compute reliability diagram values.

    For each bin :math:`b`:

    .. math::

        \text{acc}(b) = \frac{1}{|B_b|}\sum_{i \in B_b} y_i, \qquad
        \text{conf}(b) = \frac{1}{|B_b|}\sum_{i \in B_b} \hat{p}_i

    Calibration gap = :math:`|\text{acc}(b) - \text{conf}(b)|`.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels ``{0, 1}``, shape ``(n,)``.
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n,)``.
    n_bins : int
        Number of bins.
    strategy : str
        ``"uniform"`` (equal-width) or ``"quantile"``
        (equal-frequency bins).

    Returns
    -------
    dict
        ``bin_centers``, ``fraction_positive``, ``mean_confidence``,
        ``bin_counts``, ``calibration_gaps``, ``overconfident_bins``,
        ``underconfident_bins``, ``n``, ``method``.

    References
    ----------
    DeGroot & Fienberg (1983). JRSS-D, 32(1-2), 12-22.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError("y_prob values must be in [0, 1].")
    n = len(y_true)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        n_bins = len(bin_edges) - 1
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    bin_centers = np.empty(n_bins)
    frac_pos = np.empty(n_bins)
    mean_conf = np.empty(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        counts[i] = int(mask.sum())
        bin_centers[i] = (lo + hi) / 2.0
        if counts[i] > 0:
            frac_pos[i] = float(y_true[mask].mean())
            mean_conf[i] = float(y_prob[mask].mean())
        else:
            frac_pos[i] = float("nan")
            mean_conf[i] = float("nan")

    gaps = np.abs(frac_pos - mean_conf)
    overconf = np.where(mean_conf > frac_pos)[0].tolist()
    underconf = np.where(mean_conf < frac_pos)[0].tolist()

    return {
        "bin_centers": bin_centers,
        "fraction_positive": frac_pos,
        "mean_confidence": mean_conf,
        "bin_counts": counts,
        "calibration_gaps": gaps,
        "overconfident_bins": overconf,
        "underconfident_bins": underconf,
        "n": n,
        "method": f"reliability-diagram-{strategy}",
    }


def cheatsheet() -> str:
    return "reldi(y_true, y_prob) -> Reliability diagram (DeGroot & Fienberg 1983, JRSS-D)."
