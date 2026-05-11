# morie.fn — function file (hadesllm/morie)
"""Nested CFA model comparison (chi-square difference test)."""

from __future__ import annotations

from scipy import stats as sp


def cfa_compare(fit1: dict, fit2: dict, cdf=None, *, alpha: float = 0.05) -> dict:
    """Compare two nested CFA models via chi-square difference test.

    The more constrained model should have higher chi2 and more df.
    Convention: fit1 = less constrained (fewer df), fit2 = more constrained.

    Parameters
    ----------
    fit1 : dict
        Fit result from the less constrained model (requires 'chi2', 'df', 'cfi').
    fit2 : dict
        Fit result from the more constrained model.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict
        Keys: delta_chi2, delta_df, p_value, delta_cfi, significant.
        significant is True if the constraint significantly worsens fit.

    References
    ----------
    Satorra, A. & Bentler, P.M. (2001). A scaled difference chi-square
        test statistic. Psychometrika, 66(4), 507-514.
    """
    chi2_1 = fit1.get("chi2", 0.0)
    chi2_2 = fit2.get("chi2", 0.0)
    df_1 = fit1.get("df", 0)
    df_2 = fit2.get("df", 0)

    delta_chi2 = abs(chi2_2 - chi2_1)
    delta_df = abs(df_2 - df_1)

    if delta_df < 1:
        delta_df = 1

    p_value = float(1 - sp.chi2.cdf(delta_chi2, delta_df))

    cfi_1 = fit1.get("cfi", 1.0)
    cfi_2 = fit2.get("cfi", 1.0)
    delta_cfi = cfi_1 - cfi_2  # positive = constrained model is worse

    return {
        "delta_chi2": float(delta_chi2),
        "delta_df": int(delta_df),
        "p_value": float(p_value),
        "delta_cfi": float(delta_cfi),
        "significant": p_value < alpha,
    }


def cheatsheet() -> str:
    return "cfa_compare({}) -> Nested CFA model comparison (chi-square difference test)."
