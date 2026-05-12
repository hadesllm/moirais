# morie.fn -- function file (hadesllm/morie)
"""Cohen's kappa inter-rater agreement."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def cohens_kappa(rater1, rater2) -> ESRes:
    """Cohen's kappa for inter-rater agreement.

    Parameters
    ----------
    rater1, rater2 : array-like
        Categorical ratings from two raters. Must have the same length.

    Returns
    -------
    ESRes
    """
    r1 = np.asarray(rater1).ravel()
    r2 = np.asarray(rater2).ravel()
    if len(r1) != len(r2):
        raise ValueError("Raters must have same length")
    n = len(r1)
    categories = np.unique(np.concatenate([r1, r2]))
    k = len(categories)
    # Confusion matrix
    conf = np.zeros((k, k), dtype=int)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    for a, b in zip(r1, r2):
        conf[cat_to_idx[a], cat_to_idx[b]] += 1
    po = float(np.trace(conf)) / n
    pe = float(np.sum(conf.sum(0) * conf.sum(1))) / (n * n)
    kap = (po - pe) / (1 - pe) if pe < 1 else 0.0
    # SE (Fleiss, 1971)
    se = float(np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))) if pe < 1 else 0.0
    z = stats.norm.ppf(0.975)
    return ESRes(
        measure="Cohen's kappa",
        estimate=float(kap),
        ci_lower=kap - z * se,
        ci_upper=kap + z * se,
        se=se,
        n=n,
    )


kappa = cohens_kappa


def cheatsheet() -> str:
    return "cohens_kappa({}) -> Cohen's kappa inter-rater agreement."
