# morie.fn -- function file (hadesllm/morie)
"""Overlap weighting for causal inference."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def overlap_weight(
    ps_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> DescriptiveResult:
    """Overlap (tilting) weights ATE estimator (Li, Morgan & Zaslavsky, 2018).

    w_i = 1 - e(x_i) for treated, e(x_i) for controls.

    Parameters
    ----------
    ps_scores : (n,) propensity scores
    treatment : (n,) binary
    outcome : (n,) outcome

    Returns
    -------
    DescriptiveResult
    """
    ps = np.asarray(ps_scores, dtype=float).ravel()
    t = np.asarray(treatment, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    n = len(ps)

    w = np.where(t == 1, 1 - ps, ps)
    w = np.clip(w, 1e-8, None)

    y1_w = np.sum(w * t * y) / (np.sum(w * t) + 1e-12)
    y0_w = np.sum(w * (1 - t) * y) / (np.sum(w * (1 - t)) + 1e-12)
    ate = float(y1_w - y0_w)

    return DescriptiveResult(
        name="overlap_weight",
        value=ate,
        extra={
            "y1_weighted_mean": float(y1_w),
            "y0_weighted_mean": float(y0_w),
            "ess_treated": float(np.sum(w * t) ** 2 / np.sum((w * t) ** 2)),
            "ess_control": float(np.sum(w * (1 - t)) ** 2 / np.sum((w * (1 - t)) ** 2)),
            "n": n,
        },
    )


ovrla = overlap_weight


def cheatsheet() -> str:
    return "overlap_weight({}) -> Overlap weighting for causal inference."
