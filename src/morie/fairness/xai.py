# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-agnostic explainability (XAI) for bias discovery.

Reimplements the explainer suite of the COMPAS audit in pbiecek's
*XAI Stories* (story-compas): permutation feature importance, partial
dependence, accumulated local effects (ALE), ceteris paribus, and
Shapley values.  Every explainer is *model-agnostic* — it takes a bare
``predict_fn`` callable, so it works on any classifier or risk model,
not just morie's own.

The bias-discovery angle: the COMPAS audit found that *race* and *sex*
ranked high in the model's feature importance — a direct fingerprint
of a discriminatory model.  :func:`xai_permutation_importance` accepts
a ``protected`` argument and flags exactly that.

Methods reimplemented from their published definitions (Fisher et al.
on permutation importance; Friedman on partial dependence; Apley &
Zhu on ALE; Štrumbelj & Kononenko / Lundberg & Lee on Shapley
sampling); no code copied.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from morie.fn._richresult import RichResult

__all__ = [
    "xai_permutation_importance",
    "xai_partial_dependence",
    "xai_ale",
    "xai_ceteris_paribus",
    "xai_shap_values",
]


def _as_2d(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be a 2-D array (n_samples, n_features)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("X is empty")
    return arr


def _names(feature_names: Any, d: int) -> list[str]:
    if feature_names is None:
        return [f"x{j}" for j in range(d)]
    names = list(feature_names)
    if len(names) != d:
        raise ValueError(
            f"feature_names has {len(names)} entries; X has {d} columns"
        )
    return names


def _resolve(feature: Any, names: list[str]) -> int:
    if isinstance(feature, str):
        if feature not in names:
            raise ValueError(f"feature {feature!r} not in {names}")
        return names.index(feature)
    idx = int(feature)
    if not (0 <= idx < len(names)):
        raise ValueError(f"feature index {idx} out of range")
    return idx


def _predict(predict_fn: Callable, X: np.ndarray) -> np.ndarray:
    out = np.asarray(predict_fn(X), dtype=float).reshape(-1)
    if out.shape[0] != X.shape[0]:
        raise ValueError(
            "predict_fn must return one prediction per row of X"
        )
    return out


def xai_permutation_importance(
    predict_fn: Callable,
    X: Any,
    *,
    feature_names: Any = None,
    n_repeats: int = 10,
    protected: Any = None,
    seed: int = 0,
) -> RichResult:
    """Permutation feature importance — model reliance on each feature.

    Importance of a feature is the mean absolute change in the model's
    predictions when that feature's column is randomly shuffled (its
    information destroyed while the marginal distribution is kept).  A
    feature the model ignores scores ≈ 0; a feature it leans on scores
    high.

    Parameters
    ----------
    predict_fn : callable
        Maps an ``(n, d)`` array to ``n`` predictions.
    X : array-like, shape (n, d)
        The data to explain over.
    feature_names : sequence of str, optional
    n_repeats : int
        Number of shuffles averaged per feature.
    protected : sequence of str, optional
        Protected-attribute feature names.  If any rank in the top
        third of features, a bias warning is raised — the COMPAS-audit
        fingerprint of a discriminatory model.
    seed : int

    Returns
    -------
    RichResult
        Headline value is the largest importance; payload carries the
        per-feature ``importances`` and the protected-feature ranks.

    Examples
    --------
    >>> import numpy as np, morie
    >>> X = np.random.default_rng(0).normal(size=(200, 3))
    >>> res = morie.xai_permutation_importance(
    ...     lambda A: A[:, 0], X, feature_names=["a", "b", "c"])
    >>> res.payload["ranking"][0]          # feature 'a' dominates
    'a'
    """
    X = _as_2d(X)
    n, d = X.shape
    names = _names(feature_names, d)
    rng = np.random.default_rng(seed)
    base = _predict(predict_fn, X)

    importances: dict[str, float] = {}
    for j in range(d):
        deltas = []
        for _ in range(int(n_repeats)):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(n), j]
            deltas.append(float(np.mean(np.abs(_predict(predict_fn, Xp)
                                                - base))))
        importances[names[j]] = float(np.mean(deltas))

    ranking = sorted(importances, key=lambda k: importances[k], reverse=True)
    top = importances[ranking[0]] if ranking else 0.0

    warnings: list[str] = []
    protected_ranks: dict[str, int] = {}
    if protected:
        for p in protected:
            if p not in importances:
                raise ValueError(f"protected feature {p!r} not in features")
            protected_ranks[p] = ranking.index(p) + 1
        flagged = [p for p, r in protected_ranks.items()
                   if r <= max(1, d // 3)]
        if flagged:
            warnings.append(
                f"protected attribute(s) {flagged} rank in the top third "
                f"of feature importance — the model leans materially on a "
                f"protected characteristic, a direct bias signal."
            )

    table = [[nm, round(importances[nm], 5),
              ("protected" if nm in protected_ranks else "")]
             for nm in ranking]
    interp = (
        f"The model relies most on {ranking[0]!r} "
        f"(importance {top:.4f}). "
        + ("Protected attributes appear high in the ranking — see the "
           "warning above." if warnings else
           "No protected attribute ranks in the top third.")
        if ranking else "No features to rank."
    )
    return RichResult(
        title="Permutation Feature Importance",
        summary_lines=[("Top feature", ranking[0] if ranking else None),
                       ("Top importance", top)],
        tables=[{"title": "Feature importance (high → low):",
                 "headers": ["feature", "importance", "flag"],
                 "rows": table}],
        warnings=warnings,
        interpretation=interp,
        payload={"value": top, "importances": importances,
                 "ranking": ranking, "protected_ranks": protected_ranks},
    )


def xai_partial_dependence(
    predict_fn: Callable,
    X: Any,
    feature: Any,
    *,
    feature_names: Any = None,
    grid_size: int = 20,
) -> RichResult:
    """Partial dependence of the model on one feature (Friedman).

    Sweeps ``feature`` across a grid; at each grid value *every* row is
    set to that value and the predictions are averaged.  The resulting
    curve is the model's average response to the feature.

    Returns
    -------
    RichResult
        payload carries the ``grid`` and ``pd`` (partial-dependence)
        arrays; headline value is the curve's range.
    """
    X = _as_2d(X)
    names = _names(feature_names, X.shape[1])
    j = _resolve(feature, names)
    lo, hi = float(np.min(X[:, j])), float(np.max(X[:, j]))
    grid = np.linspace(lo, hi, int(grid_size))
    pd_vals = []
    for v in grid:
        Xv = X.copy()
        Xv[:, j] = v
        pd_vals.append(float(np.mean(_predict(predict_fn, Xv))))
    pd_vals = np.array(pd_vals)
    rng_val = float(pd_vals.max() - pd_vals.min())
    return RichResult(
        title=f"Partial Dependence — {names[j]}",
        summary_lines=[("Feature", names[j]),
                       ("PD range", rng_val),
                       ("PD at min", float(pd_vals[0])),
                       ("PD at max", float(pd_vals[-1]))],
        interpretation=(
            f"As {names[j]!r} sweeps its observed range, the model's "
            f"average prediction moves over a span of {rng_val:.4f}."
        ),
        payload={"value": rng_val, "feature": names[j],
                 "grid": grid, "pd": pd_vals},
    )


def xai_ale(
    predict_fn: Callable,
    X: Any,
    feature: Any,
    *,
    feature_names: Any = None,
    n_bins: int = 10,
) -> RichResult:
    """First-order Accumulated Local Effects for one feature (Apley & Zhu).

    ALE bins the feature, measures the *local* prediction change across
    each bin's edges using only rows that fall in the bin, then
    accumulates and centres those local effects.  Unlike partial
    dependence it stays unbiased when features are correlated.

    Returns
    -------
    RichResult
        payload carries ``bin_centers`` and ``ale`` (centred effect).
    """
    X = _as_2d(X)
    names = _names(feature_names, X.shape[1])
    j = _resolve(feature, names)
    col = X[:, j]
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(col, qs))
    if edges.size < 2:
        raise ValueError(f"feature {names[j]!r} has no spread for ALE")
    k = edges.size - 1

    local = np.zeros(k)
    counts = np.zeros(k, dtype=int)
    bin_idx = np.clip(np.searchsorted(edges, col, side="left") - 1, 0, k - 1)
    for b in range(k):
        rows = X[bin_idx == b]
        if rows.shape[0] == 0:
            continue
        lo_rows = rows.copy()
        lo_rows[:, j] = edges[b]
        hi_rows = rows.copy()
        hi_rows[:, j] = edges[b + 1]
        diff = _predict(predict_fn, hi_rows) - _predict(predict_fn, lo_rows)
        local[b] = float(np.mean(diff))
        counts[b] = rows.shape[0]

    accumulated = np.concatenate([[0.0], np.cumsum(local)])
    # centre by the count-weighted mean over bin midpoints
    mid = 0.5 * (accumulated[:-1] + accumulated[1:])
    total = counts.sum()
    centre = float(np.sum(mid * counts) / total) if total else 0.0
    ale = accumulated - centre
    bin_centers = edges
    return RichResult(
        title=f"Accumulated Local Effects — {names[j]}",
        summary_lines=[("Feature", names[j]),
                       ("Bins", k),
                       ("ALE range", float(ale.max() - ale.min()))],
        interpretation=(
            f"The accumulated local effect of {names[j]!r} spans "
            f"{float(ale.max() - ale.min()):.4f} across its range, "
            f"correlation-robustly."
        ),
        payload={"value": float(ale.max() - ale.min()),
                 "feature": names[j], "bin_centers": bin_centers,
                 "ale": ale},
    )


def xai_ceteris_paribus(
    predict_fn: Callable,
    x: Any,
    feature: Any,
    *,
    X_ref: Any,
    feature_names: Any = None,
    grid_size: int = 20,
) -> RichResult:
    """Ceteris-paribus profile — one instance, one feature varied.

    Holds every feature of the instance ``x`` fixed except ``feature``,
    which is swept across the range seen in ``X_ref``.  This is the
    instance-level analogue of partial dependence and the basis of the
    COMPAS audit's "what if this defendant were older / a different
    race" probes.

    Returns
    -------
    RichResult
        payload carries the ``grid`` and the ``profile`` predictions.
    """
    X_ref = _as_2d(X_ref)
    names = _names(feature_names, X_ref.shape[1])
    j = _resolve(feature, names)
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != X_ref.shape[1]:
        raise ValueError("x must have one value per feature of X_ref")
    lo, hi = float(np.min(X_ref[:, j])), float(np.max(X_ref[:, j]))
    grid = np.linspace(lo, hi, int(grid_size))
    rows = np.tile(x, (grid.size, 1))
    rows[:, j] = grid
    profile = _predict(predict_fn, rows)
    base = float(_predict(predict_fn, x.reshape(1, -1))[0])
    swing = float(profile.max() - profile.min())
    return RichResult(
        title=f"Ceteris-Paribus Profile — {names[j]}",
        summary_lines=[("Feature", names[j]),
                       ("Instance prediction", base),
                       ("Profile swing", swing)],
        interpretation=(
            f"Holding this instance fixed and varying {names[j]!r} alone, "
            f"the prediction swings by {swing:.4f}. A large swing on a "
            f"protected feature means the decision would change purely on "
            f"that characteristic."
        ),
        payload={"value": swing, "feature": names[j],
                 "grid": grid, "profile": profile, "base": base},
    )


def xai_shap_values(
    predict_fn: Callable,
    x: Any,
    background: Any,
    *,
    feature_names: Any = None,
    n_samples: int = 200,
    seed: int = 0,
) -> RichResult:
    """Shapley feature attributions for one instance (sampling estimator).

    Estimates SHAP values by the permutation-sampling method
    (Štrumbelj & Kononenko; the model-agnostic explainer of Lundberg &
    Lee): over many random feature orderings and random background
    rows, a feature's value is "switched on" and its marginal effect on
    the prediction recorded.  The values are additive — they sum to the
    instance's prediction minus the mean background prediction.

    Returns
    -------
    RichResult
        Headline value is the largest-magnitude SHAP value; payload
        carries the per-feature ``shap_values``.
    """
    background = _as_2d(background)
    names = _names(feature_names, background.shape[1])
    d = background.shape[1]
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != d:
        raise ValueError("x must have one value per background feature")
    rng = np.random.default_rng(seed)
    nb = background.shape[0]

    contrib = np.zeros(d)
    for _ in range(int(n_samples)):
        perm = rng.permutation(d)
        # (nb, d+1, d): for every background row, the cumulative
        # coalitions along this permutation. Averaging the marginal
        # contributions over the *whole* background keeps the
        # local-accuracy identity (Σ SHAP = f(x) − E[f]) exact.
        rows = np.tile(background[:, None, :], (1, d + 1, 1))
        for k in range(d):
            rows[:, k + 1:, perm[k]] = x[perm[k]]
        preds = _predict(predict_fn, rows.reshape(nb * (d + 1), d))
        preds = preds.reshape(nb, d + 1)
        contrib[perm] += np.diff(preds, axis=1).mean(axis=0)
    shap = contrib / float(n_samples)

    shap_map = {names[j]: float(shap[j]) for j in range(d)}
    order = sorted(shap_map, key=lambda k: abs(shap_map[k]), reverse=True)
    top = shap_map[order[0]]
    fx = float(_predict(predict_fn, x.reshape(1, -1))[0])
    base = float(np.mean(_predict(predict_fn, background)))

    table = [[nm, round(shap_map[nm], 5)] for nm in order]
    return RichResult(
        title="SHAP Feature Attributions (sampling estimator)",
        summary_lines=[("Most influential feature", order[0]),
                       ("Its SHAP value", top),
                       ("Prediction", fx),
                       ("Background mean", base)],
        tables=[{"title": "SHAP values (by magnitude):",
                 "headers": ["feature", "SHAP value"], "rows": table}],
        interpretation=(
            f"For this instance the prediction {fx:.4f} departs from the "
            f"background mean {base:.4f}; {order[0]!r} contributes the "
            f"most ({top:+.4f}). The SHAP values sum to that departure."
        ),
        payload={"value": top, "shap_values": shap_map,
                 "ranking": order, "prediction": fx, "background_mean": base},
    )
