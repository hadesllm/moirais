# morie.fn -- function file (hadesllm/morie)
"""Mantel-Haenszel DIF detection with ETS classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn._containers import DIFResult


def difmh(data: pd.DataFrame | np.ndarray, group: np.ndarray | pd.Series | list, cdf=None, *, item_names: list[str] | None = None, n_strata: int = 5, alpha: float = 0.05, ref_group: int | str = 0) -> DIFResult:
    """Mantel-Haenszel DIF detection with ETS A/B/C classification.

    Stratifies examinees by total score, computes the MH odds ratio and
    chi-square statistic per item.  Items are classified into ETS DIF
    categories based on the MH delta metric (Holland & Thayer, 1988):
      A: |delta| < 1.0 (negligible)
      B: 1.0 <= |delta| < 1.5 (moderate)
      C: |delta| >= 1.5 (large)

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item response matrix (n x k), values 0/1.
    group : array-like
        Group membership vector (length n).  Two distinct values expected.
    item_names : list[str], optional
        Item labels.  If None, uses column names or item_0, item_1, ...
    n_strata : int
        Number of score strata (default 5).
    alpha : float
        Significance level for flagging (default 0.05).
    ref_group : int or str
        Value in group that identifies the reference group (default 0).

    Returns
    -------
    DIFResult
        method="MH", items DataFrame with columns: item, mh_or, mh_delta,
        chi2, p_value, classification. flagged = items with classification B or C.

    References
    ----------
    Holland, P. W. & Thayer, D. T. (1988). Differential item performance
    and the Mantel-Haenszel procedure. In H. Wainer & H. I. Braun (Eds.),
    Test Validity. Lawrence Erlbaum.

    Zwick, R. & Ercikan, K. (1989). Analysis of differential item functioning
    in the NAEP history assessment. Journal of Educational Measurement, 26(1).
    """
    X = np.asarray(data, dtype=np.float64)
    g = np.asarray(group).ravel()
    n, k = X.shape

    if len(g) != n:
        raise ValueError(f"group length ({len(g)}) != n ({n}).")

    X = np.where(np.isnan(X), 0.0, X)

    if item_names is None:
        if isinstance(data, pd.DataFrame):
            item_names = list(data.columns)
        else:
            item_names = [f"item_{j}" for j in range(k)]

    # Identify two groups
    unique_groups = sorted(set(g))
    if len(unique_groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(unique_groups)}: {unique_groups}")

    ref = ref_group
    foc = [ug for ug in unique_groups if ug != ref][0]
    is_ref = g == ref
    is_foc = g == foc

    # Total score for stratification (excluding the item being tested)
    total_score = X.sum(axis=1)

    # Create strata based on total score quantiles
    try:
        strata_bounds = np.percentile(total_score, np.linspace(0, 100, n_strata + 1))
        strata_bounds = np.unique(strata_bounds)
    except Exception:
        strata_bounds = np.array([total_score.min(), total_score.max()])

    strata = np.digitize(total_score, strata_bounds[1:-1])
    unique_strata = np.unique(strata)

    results = []
    flagged = []

    for j in range(k):
        # Rest score (total minus item j) for more refined stratification
        # But standard MH uses total score, so we keep that
        alpha_mh = 0.0
        beta_mh = 0.0
        chi2_num = 0.0
        chi2_var = 0.0

        for s in unique_strata:
            in_stratum = strata == s
            ref_s = in_stratum & is_ref
            foc_s = in_stratum & is_foc

            n_ref = ref_s.sum()
            n_foc = foc_s.sum()
            if n_ref == 0 or n_foc == 0:
                continue

            # 2x2 table for this stratum
            a_cell = X[ref_s, j].sum()  # ref correct
            b_cell = n_ref - a_cell  # ref incorrect
            c_cell = X[foc_s, j].sum()  # focal correct
            d_cell = n_foc - c_cell  # focal incorrect
            T = n_ref + n_foc

            if T == 0:
                continue

            alpha_mh += (a_cell * d_cell) / T
            beta_mh += (b_cell * c_cell) / T

            # MH chi-square components
            E_a = (a_cell + c_cell) * n_ref / T
            chi2_num += a_cell - E_a
            chi2_var += (a_cell + c_cell) * (b_cell + d_cell) * n_ref * n_foc / (T**2 * max(T - 1, 1))

        # MH odds ratio
        if beta_mh > 1e-10:
            mh_or = alpha_mh / beta_mh
        else:
            mh_or = np.nan

        # MH delta (2.35 * ln(MH_OR))
        if mh_or > 0 and np.isfinite(mh_or):
            mh_delta = -2.35 * np.log(mh_or)
        else:
            mh_delta = np.nan

        # MH chi-square with continuity correction
        if chi2_var > 1e-10:
            chi2_stat = (abs(chi2_num) - 0.5) ** 2 / chi2_var
        else:
            chi2_stat = 0.0

        p_value = 1.0 - sp.chi2.cdf(chi2_stat, df=1)

        # ETS classification
        abs_delta = abs(mh_delta) if np.isfinite(mh_delta) else 0.0
        if abs_delta < 1.0:
            classification = "A"
        elif abs_delta < 1.5:
            classification = "B"
        else:
            classification = "C"

        is_flagged = classification in ("B", "C") and p_value < alpha

        results.append(
            {
                "item": item_names[j],
                "mh_or": float(mh_or) if np.isfinite(mh_or) else np.nan,
                "mh_delta": float(mh_delta) if np.isfinite(mh_delta) else np.nan,
                "chi2": float(chi2_stat),
                "p_value": float(p_value),
                "classification": classification,
            }
        )

        if is_flagged:
            flagged.append(item_names[j])

    items_df = pd.DataFrame(results)

    return DIFResult(
        method="MH",
        items=items_df,
        flagged=flagged,
        group_var=str(ref_group),
    )


mh_dif = difmh


def cheatsheet() -> str:
    return "difmh({}) -> Mantel-Haenszel DIF detection with ETS classification."
