# morie.fn -- function file (hadesllm/morie)
"""Inter-rater reliability (Cohen's weighted kappa for ordinal data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp


def rirr(ratings1: np.ndarray | pd.Series, ratings2: np.ndarray | pd.Series, cdf=None, *, weights: str = "quadratic", alpha: float = 0.05) -> dict:
    """Cohen's weighted kappa for inter-rater reliability.

    Computes agreement between two raters on ordinal ratings, adjusting
    for chance agreement.  Quadratic weights (default) are standard for
    ordinal scales; linear weights penalise all disagreements equally.

    Parameters
    ----------
    ratings1 : array-like
        Ratings from rater 1 (integer or ordinal codes).
    ratings2 : array-like
        Ratings from rater 2 (same length and scale).
    weights : str
        Weight type: ``'quadratic'`` (default), ``'linear'``, or
        ``'identity'`` (unweighted kappa).
    alpha : float
        Significance level for CI (default 0.05).

    Returns
    -------
    dict
        Keys: ``kappa`` (float), ``se`` (float), ``ci_lower`` (float),
        ``ci_upper`` (float), ``p_value`` (float), ``n`` (int),
        ``weights`` (str), ``po`` (observed agreement), ``pe``
        (expected agreement).

    Raises
    ------
    ValueError
        If arrays differ in length or weight type is unrecognised.

    References
    ----------
    Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision
    for scaled disagreement or partial credit. *Psychological
    Bulletin*, 70(4), 213-220.

    Fleiss, J. L., Cohen, J., & Everitt, B. S. (1969). Large sample
    standard errors of kappa and weighted kappa. *Psychological
    Bulletin*, 72(5), 323-327.
    """
    r1 = np.asarray(ratings1).ravel()
    r2 = np.asarray(ratings2).ravel()

    if len(r1) != len(r2):
        raise ValueError(f"ratings1 and ratings2 must have same length, got {len(r1)} and {len(r2)}")

    # Drop pairs with NaN
    finite = np.isfinite(r1.astype(float)) & np.isfinite(r2.astype(float))
    r1 = r1[finite]
    r2 = r2[finite]
    n = len(r1)

    if n < 2:
        return {
            "kappa": float("nan"),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "n": n,
            "weights": weights,
            "po": float("nan"),
            "pe": float("nan"),
        }

    # Categories
    cats = np.sort(np.unique(np.concatenate([r1, r2])))
    q = len(cats)
    cat_idx = {c: i for i, c in enumerate(cats)}

    # Observed confusion matrix
    O = np.zeros((q, q), dtype=np.float64)
    for a, b in zip(r1, r2):
        O[cat_idx[a], cat_idx[b]] += 1.0
    O /= n  # normalise to proportions

    # Marginals
    row_marg = O.sum(axis=1)
    col_marg = O.sum(axis=0)

    # Expected matrix under independence
    E = np.outer(row_marg, col_marg)

    # Weight matrix
    if weights == "identity":
        W = 1.0 - np.eye(q)
        W = 1.0 - W  # agreement matrix: 1 on diagonal, 0 off
    elif weights == "linear":
        W = np.zeros((q, q))
        for i in range(q):
            for j in range(q):
                W[i, j] = 1.0 - abs(i - j) / max(q - 1, 1)
    elif weights == "quadratic":
        W = np.zeros((q, q))
        for i in range(q):
            for j in range(q):
                W[i, j] = 1.0 - (i - j) ** 2 / max((q - 1) ** 2, 1)
    else:
        raise ValueError(f"Unrecognised weight type '{weights}'. Use 'quadratic', 'linear', or 'identity'.")

    # Weighted observed and expected agreement
    po = np.sum(W * O)
    pe = np.sum(W * E)

    if abs(1.0 - pe) < 1e-15:
        kappa = 1.0 if abs(1.0 - po) < 1e-15 else float("nan")
        return {
            "kappa": float(kappa),
            "se": 0.0,
            "ci_lower": float(kappa),
            "ci_upper": float(kappa),
            "p_value": 0.0,
            "n": n,
            "weights": weights,
            "po": float(po),
            "pe": float(pe),
        }

    kappa = (po - pe) / (1.0 - pe)

    # Large-sample SE (Fleiss, Cohen, & Everitt, 1969)
    # SE^2 = (1 / (n * (1 - pe)^4)) * [sum_ij O_ij * (W_ij*(1-pe) - (W_i. + W_.j)*(1-po))^2 - (po*pe - 2*pe + po)^2]
    # Simplified: use the formula from Fleiss et al.
    W_row = W @ col_marg  # W_i. = sum_j w_ij * p_.j
    W_col = W.T @ row_marg  # W_.j = sum_i w_ij * p_i.

    var_sum = 0.0
    for i in range(q):
        for j in range(q):
            term = W[i, j] - (W_row[i] + W_col[j]) * (1.0 - kappa)
            var_sum += O[i, j] * term**2

    se_sq = (var_sum - (kappa - pe * (1.0 - kappa)) ** 2) / (n * (1.0 - pe) ** 2)
    se = np.sqrt(max(se_sq, 0.0))

    z_crit = sp.norm.ppf(1.0 - alpha / 2.0)
    ci_lo = kappa - z_crit * se
    ci_hi = kappa + z_crit * se

    # Two-sided test: H0: kappa = 0
    z_stat = kappa / se if se > 1e-15 else float("inf")
    p_value = 2.0 * (1.0 - sp.norm.cdf(abs(z_stat)))

    return {
        "kappa": float(kappa),
        "se": float(se),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "p_value": float(p_value),
        "n": n,
        "weights": weights,
        "po": float(po),
        "pe": float(pe),
    }


short = rirr


def cheatsheet() -> str:
    return "rirr({}) -> Inter-rater reliability (Cohen's weighted kappa for ordinal "
