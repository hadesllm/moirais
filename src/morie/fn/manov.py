# morie.fn — function file (hadesllm/morie)
"""MANOVA (Pillai/Wilks/Hotelling/Roy)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def manova(data: np.ndarray, groups: np.ndarray, cdf=None) -> DescriptiveResult:
    """One-way MANOVA with four test statistics.

    Parameters
    ----------
    data : ndarray (n, p)
        Dependent variable matrix.
    groups : ndarray (n,)
        Group labels.

    Returns
    -------
    DescriptiveResult
        ``value`` is Wilks' Lambda.
        ``extra`` has ``pillai``, ``wilks``, ``hotelling_lawley``, ``roy``,
        ``F_wilks``, ``p_wilks``, ``df1``, ``df2``.
    """
    X = np.asarray(data, dtype=np.float64)
    g = np.asarray(groups)
    n, p = X.shape
    classes = np.unique(g)
    k = len(classes)

    overall_mean = X.mean(axis=0)
    Sw = np.zeros((p, p))
    Sb = np.zeros((p, p))

    for c in classes:
        Xc = X[g == c]
        nc = Xc.shape[0]
        mc = Xc.mean(axis=0)
        diff = Xc - mc
        Sw += diff.T @ diff
        d = (mc - overall_mean).reshape(-1, 1)
        Sb += nc * (d @ d.T)

    Sw += np.eye(p) * 1e-10
    E_inv_H = np.linalg.inv(Sw) @ Sb
    eigvals = np.real(np.linalg.eigvals(E_inv_H))
    eigvals = np.sort(eigvals)[::-1]
    s = min(k - 1, p)
    lam = eigvals[:s]
    lam = np.maximum(lam, 0)

    pillai = float(np.sum(lam / (1 + lam)))
    wilks = float(np.prod(1 / (1 + lam)))
    hotelling_lawley = float(np.sum(lam))
    roy = float(lam[0]) if len(lam) > 0 else 0.0

    df1 = p * (k - 1)
    df2 = max(n - k - p + 1, 1)
    if wilks > 0:
        num = p ** 2 * (k - 1) ** 2 - 4
        den = p ** 2 + (k - 1) ** 2 - 5
        if den == 0 or num < 0:
            t = 1.0
        else:
            t = float(np.sqrt(max(num / den, 1)))
        lam_t = wilks ** (1 / t) if t > 0 else wilks
        df2_approx = max((n - 1 - (p + k) / 2) * t - (df1 - 2) / 2, 1)
        F_val = max((1 - lam_t) / lam_t * df2_approx / df1, 0) if lam_t > 0 else 0
        p_val = float(1 - sp_stats.f.cdf(F_val, df1, df2_approx))
    else:
        F_val = float("inf")
        p_val = 0.0

    return DescriptiveResult(
        name="MANOVA",
        value=wilks,
        extra={
            "pillai": pillai,
            "wilks": wilks,
            "hotelling_lawley": hotelling_lawley,
            "roy": roy,
            "F_wilks": float(F_val),
            "p_wilks": p_val,
            "df1": df1,
            "df2": df2,
            "eigenvalues": lam.tolist(),
        },
    )


manov = manova


def cheatsheet() -> str:
    return "manova({}) -> MANOVA (Pillai/Wilks/Hotelling-Lawley/Roy)."
