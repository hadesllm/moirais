# morie.fn -- function file (hadesllm/morie)
"""Propensity score matching (nearest neighbor)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def propensity_match(
    treatment: np.ndarray,
    covariates: np.ndarray,
    *,
    caliper: float | None = None,
    seed: int | None = None,
) -> DescriptiveResult:
    """Nearest-neighbor 1:1 propensity score matching.

    Estimates propensity via logistic approximation (OLS on logit-scale
    features for simplicity; no sklearn needed).

    Parameters
    ----------
    treatment : (n,) binary
    covariates : (n, p) array
    caliper : float, optional
        Max PS distance for match.
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    t = np.asarray(treatment, dtype=float).ravel()
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(t)

    X_int = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_int, t, rcond=None)[0]
    logit = X_int @ beta
    ps = 1 / (1 + np.exp(-logit))

    treated = np.where(t == 1)[0]
    control = np.where(t == 0)[0]

    matched_t = []
    matched_c = []
    used = set()
    for ti in treated:
        dists = np.abs(ps[ti] - ps[control])
        order = np.argsort(dists)
        for idx in order:
            ci = control[idx]
            if ci not in used:
                if caliper is None or dists[idx] <= caliper:
                    matched_t.append(ti)
                    matched_c.append(ci)
                    used.add(ci)
                break

    n_matched = len(matched_t)
    if n_matched > 0:
        ps_diff = np.mean(np.abs(ps[matched_t] - ps[matched_c]))
    else:
        ps_diff = float("nan")

    return DescriptiveResult(
        name="ps_match",
        value=float(n_matched),
        extra={
            "n_treated": int(len(treated)),
            "n_control": int(len(control)),
            "n_matched": n_matched,
            "mean_ps_diff": float(ps_diff),
            "caliper": caliper,
        },
    )


psmch = propensity_match


def cheatsheet() -> str:
    return "propensity_match({}) -> Propensity score matching (nearest neighbor)."
