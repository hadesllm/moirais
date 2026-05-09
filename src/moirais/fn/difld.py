# moirais.fn — function file (hadesllm/moirais)
"""Lord's chi-square DIF detection."""

from __future__ import annotations

import pandas as pd
from scipy import stats as sp

from moirais.fn._containers import DIFResult


def dif_lord_chisq(item_params_ref: dict, item_params_focal: dict, cdf=None, *, alpha: float = 0.05) -> DIFResult:
    """Lord's chi-square test for DIF.

    Compares item parameters between reference and focal groups
    using a chi-square statistic based on parameter differences.

    Parameters
    ----------
    item_params_ref : dict
        {item: {"a": float, "b": float}} for reference group.
    item_params_focal : dict
        Same for focal group.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    DIFResult
        method="Lord", items DataFrame, flagged items.

    References
    ----------
    Lord, F. M. (1980). Applications of Item Response Theory to
    Practical Testing Problems. Lawrence Erlbaum.
    """
    common = sorted(set(item_params_ref) & set(item_params_focal))
    if not common:
        raise ValueError("No common items between forms.")

    rows = []
    flagged = []
    for item in common:
        ref = item_params_ref[item]
        foc = item_params_focal[item]
        da = ref.get("a", 1.0) - foc.get("a", 1.0)
        db = ref.get("b", 0.0) - foc.get("b", 0.0)
        chi2 = da**2 + db**2
        df = 2
        p_val = 1.0 - sp.chi2.cdf(chi2, df)
        sig = p_val < alpha
        rows.append(
            {
                "item": item,
                "chi2": float(chi2),
                "df": df,
                "p_value": float(p_val),
                "delta_a": float(da),
                "delta_b": float(db),
            }
        )
        if sig:
            flagged.append(item)

    return DIFResult(method="Lord", items=pd.DataFrame(rows), flagged=flagged)


lord_dif = dif_lord_chisq


def cheatsheet() -> str:
    return "dif_lord_chisq({}) -> Lord's chi-square DIF detection."
