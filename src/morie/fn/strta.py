"""Stratified analysis (2x2xK) with Breslow-Day test."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def stratified_analysis(tables: list[tuple[int, int, int, int]], confidence: float = 0.95, cdf=None) -> ESRes:
    """Stratified 2x2xK analysis with homogeneity test.

    Computes stratum-specific ORs, Mantel-Haenszel pooled OR,
    and Breslow-Day test for homogeneity of ORs.

    Parameters
    ----------
    tables : list of (a, b, c, d) tuples
        Each tuple is a 2x2 table stratum.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Breslow, N. E. & Day, N. E. (1980). *Statistical Methods in
    Cancer Research*, Vol. I. IARC Scientific Publications No. 32.
    """
    if len(tables) < 2:
        raise ValueError("Need at least 2 strata")

    stratum_ors = []
    z = stats.norm.ppf((1 + confidence) / 2)
    num_mh = 0.0
    den_mh = 0.0

    for a, b, c, d in tables:
        n_k = a + b + c + d
        if b * c > 0:
            or_k = (a * d) / (b * c)
        else:
            or_k = np.inf
        stratum_ors.append(or_k)
        num_mh += a * d / n_k
        den_mh += b * c / n_k

    or_mh = num_mh / den_mh if den_mh > 0 else np.inf

    bd_stat = 0.0
    for a, b, c, d in tables:
        n_k = a + b + c + d
        m1 = a + b
        m0 = c + d
        n1 = a + c
        n0 = b + d

        aa = 1 - 1 / or_mh
        bb = -(m1 + n1 * or_mh + (1 - or_mh))
        cc = n1 * m1 * or_mh

        if abs(aa) < 1e-15:
            a_exp = cc / (-bb) if abs(bb) > 0 else a
        else:
            disc = bb**2 - 4 * aa * cc
            a_exp = (-bb - np.sqrt(max(disc, 0))) / (2 * aa)

        a_exp = max(0.5, min(a_exp, min(m1, n1) - 0.5))
        b_exp = m1 - a_exp
        c_exp = n1 - a_exp
        d_exp = m0 - c_exp

        var_a = 1.0 / (1 / max(a_exp, 0.5) + 1 / max(b_exp, 0.5) +
                        1 / max(c_exp, 0.5) + 1 / max(d_exp, 0.5))
        bd_stat += (a - a_exp) ** 2 / var_a

    bd_df = len(tables) - 1
    bd_p = 1 - stats.chi2.cdf(bd_stat, bd_df)

    return ESRes(
        measure="stratified_2x2xK",
        estimate=float(or_mh),
        n=sum(a + b + c + d for a, b, c, d in tables),
        extra={
            "stratum_ORs": [float(x) for x in stratum_ors],
            "OR_MH": float(or_mh),
            "breslow_day_chi2": float(bd_stat),
            "breslow_day_df": bd_df,
            "breslow_day_p": float(bd_p),
            "homogeneous": bd_p > 0.05,
        },
    )


strta = stratified_analysis


def cheatsheet() -> str:
    return "stratified_analysis({}) -> Stratified 2x2xK with Breslow-Day homogeneity test."
