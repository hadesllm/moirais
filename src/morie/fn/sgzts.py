"""Asymptotic z-test for spatial statistics."""

from __future__ import annotations

from ._containers import DescriptiveResult


def asymptotic_z_test(observed_stat, expected, se, cdf=None):
    """Asymptotic z-test for a spatial statistic.

    .. epigraph:: "Do a barrel roll!" -- Peppy Hare, Star Fox

    Parameters
    ----------
    observed_stat : float
        Observed value of the statistic.
    expected : float
        Expected value under the null.
    se : float
        Standard error of the statistic.

    Returns
    -------
    DescriptiveResult
    """
    from scipy import stats

    z = (observed_stat - expected) / se if se > 0 else 0.0
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return DescriptiveResult(
        name="asymptotic_z_test",
        value=float(z),
        extra={
            "z_statistic": float(z),
            "p_value": float(p_value),
            "observed": float(observed_stat),
            "expected": float(expected),
            "se": float(se),
            "significant": p_value < 0.05,
        },
    )


sgzts = asymptotic_z_test


def cheatsheet() -> str:
    return "asymptotic_z_test({}) -> Asymptotic z-test for spatial statistics."
