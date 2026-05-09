# moirais.fn — function file (hadesllm/moirais)
"""One-way MANOVA (Wilks' lambda)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def manova_one(X: np.ndarray, groups: np.ndarray, cdf=None) -> TestResult:
    """One-way MANOVA using Wilks' lambda.

    Parameters
    ----------
    X : (n, p) array
    groups : (n,) group labels

    Returns
    -------
    TestResult
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    labels = np.unique(groups)
    k = len(labels)
    if k < 2:
        raise ValueError("Need >= 2 groups.")

    grand_mean = X.mean(axis=0)
    W = np.zeros((p, p))
    B = np.zeros((p, p))
    for lab in labels:
        Xi = X[groups == lab]
        ni = len(Xi)
        mi = Xi.mean(axis=0)
        d = mi - grand_mean
        B += ni * np.outer(d, d)
        W += (Xi - mi).T @ (Xi - mi)

    try:
        lam = np.linalg.det(W) / np.linalg.det(W + B)
    except np.linalg.LinAlgError:
        lam = 1.0

    df_h = p * (k - 1)
    df_e = p * (n - k)
    t_val = np.sqrt((p**2 * (k - 1) ** 2 - 4) / (p**2 + (k - 1) ** 2 - 5)) if (p**2 + (k - 1) ** 2 - 5) > 0 else 1.0
    lam_t = max(lam, 1e-12) ** (1 / t_val)
    df2 = (n - k - 1) - (p - (k - 1) + 1) / 2
    df2 = max(df2, 1)
    F_approx = (1 - lam_t) / lam_t * df2 / df_h if df_h > 0 else 0.0
    p_val = float(1 - sp_stats.f.cdf(F_approx, df_h, df2))

    return TestResult(
        test_name="MANOVA",
        statistic=float(lam),
        p_value=p_val,
        df=float(df_h),
        method="Wilks lambda",
        n=n,
        extra={"F_approx": float(F_approx), "k": k, "p": p},
    )


manova = manova_one


def cheatsheet() -> str:
    return "manova_one({}) -> One-way MANOVA (Wilks' lambda)."
