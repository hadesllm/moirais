# moirais.fn — function file (hadesllm/moirais)
"""Omega-squared from ANOVA -- less biased than eta-squared."""


def omega_squared(f_stat: float, df_between: int, df_within: int, n: int) -> float:
    """
    Omega-squared (omega^2) from ANOVA -- less biased than eta-squared.

    omega^2 = (SS_between - df_between * MS_within) / (SS_total + MS_within)
            = (df_between * (F - 1)) / (df_between * F + df_within + 1)

    :param f_stat: F-statistic from one-way ANOVA.
    :param df_between: Between-groups df (k - 1).
    :param df_within: Within-groups df (N - k).
    :param n: Total sample size N.
    :return: Omega-squared; clipped to [0, 1] as negative values are set to 0.
    :raises ValueError: If f_stat < 0 or df values are non-positive.

    References
    ----------
    Hays, W. L. (1963). Statistics for Psychologists. Holt, Rinehart and Winston.
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    if f_stat < 0:
        raise ValueError(f"f_stat must be >= 0, got {f_stat}.")
    if df_between <= 0 or df_within <= 0:
        raise ValueError("df_between and df_within must be > 0.")
    num = df_between * (f_stat - 1)
    denom = df_between * f_stat + df_within + 1
    return float(max(0.0, num / denom)) if denom > 0 else 0.0


omega2 = omega_squared


def cheatsheet() -> str:
    return "omega_squared({}) -> Omega-squared from ANOVA -- less biased than eta-squared."
