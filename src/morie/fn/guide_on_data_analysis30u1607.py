"""GeneralStatistics expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_30_unnumbered_1607"]


def guide_on_data_analysis_chapter_30_unnumbered_1607(x, cdf=None):
    """
    GeneralStatistics expression (auto-extracted; see ref).

    Formula: [EQ] 1 + [𝔼(𝜇1,1(𝑋)−𝜇0,1(𝑋) ∣ 𝐷 = 1) − 𝔼(𝜇1,1(𝑋)−𝜇0,1(𝑋) ∣ 𝐷 = 1, 𝑇 = 1)] − [𝔼(𝜇1,0(𝑋)−𝜇0,0(𝑋) ∣ 𝐷 = 1) − 𝔼(𝜇1,0(𝑋)−𝜇0,0(𝑋) ∣ 𝐷 = 1, 𝑇 = 0)]

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
        See ``morie.fn.describe('guide_on_data_analysis30u1607')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.30 (unnumbered)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return hypothesis_test_result(
            test_name="GeneralStatistics expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<2: insufficient data."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "GeneralStatistics expression (auto-extracted; see ref).", "p_value": float("nan")},
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
        test_name="GeneralStatistics expression (auto-extracted; see ref).",
        statistic=float(statistic),
        pvalue=float(p_value),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "GeneralStatistics expression (auto-extracted; see ref).", "p_value": float(p_value)},
    )


def cheatsheet():
    return "guide_on_data_analysis30u1607: GeneralStatistics expression (auto-extracted; see ref)."
