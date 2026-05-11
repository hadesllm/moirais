# morie.fn — function file (hadesllm/morie)
"""Propensity score stratification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def propensity_stratify(
    ps_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_strata: int = 5,
) -> DescriptiveResult:
    """Propensity score stratification (subclassification).

    Parameters
    ----------
    ps_scores : (n,) propensity scores
    treatment : (n,) binary
    outcome : (n,) outcome
    n_strata : int

    Returns
    -------
    DescriptiveResult
    """
    ps = np.asarray(ps_scores, dtype=float).ravel()
    t = np.asarray(treatment, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    n = len(ps)

    boundaries = np.percentile(ps, np.linspace(0, 100, n_strata + 1))
    boundaries[0] -= 1e-6
    boundaries[-1] += 1e-6

    strata_effects = []
    strata_sizes = []
    for s in range(n_strata):
        mask = (ps > boundaries[s]) & (ps <= boundaries[s + 1])
        ys_t = y[mask & (t == 1)]
        ys_c = y[mask & (t == 0)]
        if len(ys_t) > 0 and len(ys_c) > 0:
            strata_effects.append(float(ys_t.mean() - ys_c.mean()))
        else:
            strata_effects.append(float("nan"))
        strata_sizes.append(int(mask.sum()))

    valid = [e for e in strata_effects if np.isfinite(e)]
    ate = float(np.mean(valid)) if valid else float("nan")

    return DescriptiveResult(
        name="ps_stratify",
        value=ate,
        extra={"strata_effects": strata_effects, "strata_sizes": strata_sizes, "n_strata": n_strata, "n": n},
    )


psstr = propensity_stratify


def cheatsheet() -> str:
    return "propensity_stratify({}) -> Propensity score stratification."
