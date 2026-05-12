# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Anchor explanations (rule-based local explanations).

Finds a minimal set of feature conditions (an anchor) such that the
model prediction is sufficiently stable when those conditions hold.

References
----------
Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). Anchors: High-
precision model-agnostic explanations.
*Proceedings of AAAI*, 32(1), 1527-1535.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["anchr"]

_N_BINS = 4
_PRECISION_THRESHOLD = 0.95


def anchr(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    X_train: np.ndarray,
    *,
    precision_threshold: float = _PRECISION_THRESHOLD,
    n_samples_per_rule: int = 200,
    max_anchor_size: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    r"""Find an anchor explanation for a single prediction.

    An anchor :math:`A` is a set of predicates such that:

    .. math::

        P(f(z) = f(x) \mid A(z) = 1) \geq \tau

    where :math:`z` is drawn from the neighbourhood of :math:`x`
    and :math:`\tau` is ``precision_threshold``.

    Uses a greedy beam search over discretised feature conditions.

    Parameters
    ----------
    predict_fn : Callable
        Model function ``(n, p) -> (n,)`` returning predictions.
    instance : np.ndarray
        Instance to explain, shape ``(p,)``.
    X_train : np.ndarray
        Training data for discretisation and sampling, shape ``(n, p)``.
    precision_threshold : float
        Minimum precision :math:`\tau` for the anchor.
    n_samples_per_rule : int
        Samples to evaluate each candidate rule.
    max_anchor_size : int
        Maximum number of predicates in the anchor.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``anchor_features`` (list of feature indices in anchor),
        ``anchor_conditions`` (list of (feature, bin_lo, bin_hi)),
        ``precision``, ``coverage``, ``prediction``,
        ``p``, ``method``.

    References
    ----------
    Ribeiro et al. (2018). AAAI 32(1), 1527-1535.
    """
    instance = np.asarray(instance, dtype=float).ravel()
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2-D.")
    p = len(instance)
    n_train = X_train.shape[0]
    rng = np.random.default_rng(seed)

    # Get prediction for the instance
    pred_instance = float(predict_fn(instance[None, :])[0])

    # Discretise features into quantile bins
    bins = []
    for j in range(p):
        quantiles = np.quantile(X_train[:, j], np.linspace(0, 1, _N_BINS + 1))
        quantiles = np.unique(quantiles)
        bins.append(quantiles)

    def _get_bin(j, val):
        b = bins[j]
        idx = int(np.searchsorted(b, val, side="right")) - 1
        return max(0, min(idx, len(b) - 2))

    instance_bins = [_get_bin(j, instance[j]) for j in range(p)]

    def _eval_rule(feature_set):
        """Estimate precision and coverage of a rule."""
        if not feature_set:
            return 0.0, 1.0
        # Sample from training data satisfying the rule
        satisfies = np.ones(n_train, dtype=bool)
        for j in feature_set:
            b = bins[j]
            bi = instance_bins[j]
            lo = b[bi]
            hi = b[bi + 1] if bi + 1 < len(b) else b[-1] + 1e-8
            satisfies &= (X_train[:, j] >= lo) & (X_train[:, j] <= hi)
        if satisfies.sum() == 0:
            return 0.0, 0.0
        # Sample from satisfying region
        idx = np.where(satisfies)[0]
        sample_idx = rng.choice(idx, size=min(n_samples_per_rule, len(idx)), replace=True)
        X_sample = X_train[sample_idx].copy()
        # Replace non-anchor features with instance values to stay local
        for j in range(p):
            if j not in feature_set:
                X_sample[:, j] = instance[j] + rng.normal(0, 0.01, size=len(sample_idx))
        preds = predict_fn(X_sample)
        if hasattr(preds[0], '__round__'):
            same = np.mean(np.round(preds) == round(pred_instance))
        else:
            same = np.mean(np.abs(preds - pred_instance) < 0.5)
        coverage = float(satisfies.sum() / n_train)
        return float(same), coverage

    # Greedy beam search
    best_anchor: list[int] = []
    best_precision = 0.0
    best_coverage = 0.0

    candidates: list[list[int]] = [[j] for j in range(p)]
    for _ in range(max_anchor_size):
        new_candidates = []
        for feat_set in candidates:
            prec, cov = _eval_rule(feat_set)
            if prec >= precision_threshold:
                if len(feat_set) < len(best_anchor) or best_precision < precision_threshold:
                    best_anchor = feat_set
                    best_precision = prec
                    best_coverage = cov
            else:
                # Expand
                for j in range(p):
                    if j not in feat_set:
                        new_candidates.append(feat_set + [j])
        if best_precision >= precision_threshold:
            break
        candidates = new_candidates
        if not candidates:
            break

    # Build conditions
    conditions = []
    for j in best_anchor:
        b = bins[j]
        bi = instance_bins[j]
        lo = float(b[bi])
        hi = float(b[bi + 1]) if bi + 1 < len(b) else float(b[-1])
        conditions.append({"feature": j, "lo": lo, "hi": hi})

    return {
        "anchor_features": best_anchor,
        "anchor_conditions": conditions,
        "precision": best_precision,
        "coverage": best_coverage,
        "prediction": pred_instance,
        "p": p,
        "method": "Anchor",
    }


def cheatsheet() -> str:
    return "anchr(predict_fn, instance, X_train) -> Anchor rule explanation (Ribeiro et al. 2018, AAAI)."
