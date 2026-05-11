# morie.fn — function file (hadesllm/morie)
"""DIF bundle/testlet analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn._containers import DIFResult


def dif_bundle(responses: np.ndarray | pd.DataFrame, group: np.ndarray | list, bundles: dict[str, list[int]], cdf=None, *, alpha: float = 0.05) -> DIFResult:
    """Bundle-level DIF analysis for item testlets.

    Tests whether bundles of items show DIF collectively using
    a z-test on the mean difference in bundle scores.

    Parameters
    ----------
    responses : ndarray or DataFrame
        Binary response matrix (n x k).
    group : array-like
        Group variable (two values).
    bundles : dict
        {bundle_name: [item_indices]} mapping bundles to column indices.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    DIFResult
        method="Bundle".

    References
    ----------
    Douglas, J. A., Roussos, L. A., & Stout, W. (1996). Item-bundle
    DIF hypothesis testing: Identifying suspect bundles and assessing
    their differential functioning. Journal of Educational Measurement.
    """
    X = np.asarray(responses, dtype=np.float64)
    g = np.asarray(group).ravel()
    n, k = X.shape
    X = np.where(np.isnan(X), 0, X)

    groups = sorted(set(g))
    if len(groups) != 2:
        raise ValueError("Need exactly 2 groups.")

    rows = []
    flagged = []
    for bname, indices in bundles.items():
        idx = np.array(indices)
        score_ref = X[g == groups[0]][:, idx].sum(axis=1)
        score_foc = X[g == groups[1]][:, idx].sum(axis=1)
        n_ref = len(score_ref)
        n_foc = len(score_foc)
        mean_diff = score_ref.mean() - score_foc.mean()
        pooled_var = score_ref.var(ddof=1) / max(n_ref, 1) + score_foc.var(ddof=1) / max(n_foc, 1)
        z = mean_diff / max(np.sqrt(pooled_var), 1e-10)
        p_val = 2 * (1 - sp.norm.cdf(abs(z)))

        rows.append(
            {
                "bundle": bname,
                "n_items": len(idx),
                "mean_diff": float(mean_diff),
                "z": float(z),
                "p_value": float(p_val),
            }
        )
        if p_val < alpha:
            flagged.append(bname)

    return DIFResult(method="Bundle", items=pd.DataFrame(rows), flagged=flagged)


bundle_dif = dif_bundle


def cheatsheet() -> str:
    return "dif_bundle({}) -> DIF bundle/testlet analysis."
