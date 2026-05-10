"""Nonparametric equation extracted from Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["density_estimation_for_statistics_and_data_analysis_silverma_chapter_3_equation_65"]


def density_estimation_for_statistics_and_data_analysis_silverma_chapter_3_equation_65(x, cdf=None):
    r"""
    Nonparametric equation extracted from Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability.

    Formula: [EQ] - ih h ^ r (x )A ^ h ^ t^ f\x ) + - (3.65)

    Parameters
    ----------
    x : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``moirais.fn.describe('density_estimation_for_statistics_and_data_analysis_silverma3e65')`` for the full guide.

    References
    ----------
    Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability, ch.3 eq.3.65
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return hypothesis_test_result(
            test_name="Nonparametric equation extracted from Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability.",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<2: insufficient data."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Nonparametric equation extracted from Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability.", "p_value": float("nan")},
        )
    x_sorted = np.sort(x)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(x), scale=np.std(x, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return hypothesis_test_result(
        test_name="Nonparametric equation extracted from Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability.",
        statistic=float(statistic),
        pvalue=float(p_value),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Nonparametric equation extracted from Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability.", "p_value": float(p_value)},
    )


def cheatsheet():
    return "density_estimation_for_statistics_and_data_analysis_silverma3e65: Nonparametric equation extracted from Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability."
