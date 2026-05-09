# moirais.fn — function file (hadesllm/moirais)
"""Eta-squared from ANOVA F-statistic."""


def eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    """
    Eta-squared (eta^2) from ANOVA F-statistic.

    eta^2 = SS_between / SS_total = (df_between * F) / (df_between * F + df_within)

    Conventional benchmarks: 0.01 small, 0.06 medium, 0.14 large (Cohen, 1988).

    :param f_stat: F-statistic from one-way ANOVA (>= 0).
    :param df_between: Between-groups degrees of freedom (k - 1).
    :param df_within: Within-groups degrees of freedom (N - k).
    :return: Eta-squared in [0, 1].
    :raises ValueError: If f_stat < 0 or df values are non-positive.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    if f_stat < 0:
        raise ValueError(f"f_stat must be >= 0, got {f_stat}.")
    if df_between <= 0 or df_within <= 0:
        raise ValueError("df_between and df_within must be > 0.")
    denom = df_between * f_stat + df_within
    return float(df_between * f_stat / denom) if denom > 0 else 0.0


eta2 = eta_squared


def cheatsheet() -> str:
    return "eta_squared({}) -> Eta-squared from ANOVA F-statistic."
