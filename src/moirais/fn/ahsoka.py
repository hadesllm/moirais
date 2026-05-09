# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Two-way ANOVA. 'Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu' -- Ahsoka Tano"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from ._containers import TestResult
from ._helpers import _validate_df


def anova_twoway(data: pd.DataFrame, cdf=None, *, y: str = "y", a: str = "a", b: str = "b") -> TestResult:
    """Two-way ANOVA using sum-of-squares decomposition (Type I).

    Tests main effects of factors *a* and *b* on outcome *y*.
    Dataset-agnostic: column names are keyword parameters.

    :param data: DataFrame with outcome and two factor columns.
    :param y: Name of the outcome column.
    :param a: Name of the first factor column.
    :param b: Name of the second factor column.
    :return: TestResult for factor A (factor B stats in ``extra``).
    """
    _validate_df(data, y, a, b)
    df = data[[y, a, b]].dropna()
    grand_mean = df[y].mean()
    n = len(df)

    # Factor A SS
    groups_a = df.groupby(a)[y]
    ss_a = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups_a)
    df_a = df[a].nunique() - 1

    # Factor B SS
    groups_b = df.groupby(b)[y]
    ss_b = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups_b)
    df_b = df[b].nunique() - 1

    # Total SS and residual
    ss_total = ((df[y] - grand_mean) ** 2).sum()
    ss_resid = ss_total - ss_a - ss_b
    df_resid = n - df_a - df_b - 1
    if df_resid <= 0:
        raise ValueError("Not enough observations for two-way ANOVA")

    ms_a = ss_a / df_a if df_a > 0 else 0
    ms_resid = ss_resid / df_resid

    f_stat = ms_a / ms_resid if ms_resid > 0 else float("inf")
    p_val = 1 - stats.f.cdf(f_stat, df_a, df_resid)

    # Factor B F and p
    f_b = (ss_b / df_b) / ms_resid if df_b > 0 and ms_resid > 0 else 0
    p_b = 1 - stats.f.cdf(f_b, df_b, df_resid) if df_b > 0 and ms_resid > 0 else 1.0

    return TestResult(
        test_name="Two-way ANOVA (factor A)",
        statistic=float(f_stat),
        p_value=float(p_val),
        df=float(df_a),
        n=n,
        method="Two-way ANOVA Type I",
        extra={
            "ss_a": float(ss_a),
            "ss_b": float(ss_b),
            "ss_resid": float(ss_resid),
            "f_b": float(f_b),
            "p_b": float(p_b),
        },
    )


ahsoka = anova_twoway


def cheatsheet() -> str:
    return "anova_twoway({}) -> Two-way ANOVA. 'Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu' -- Ahsoka Tano"
