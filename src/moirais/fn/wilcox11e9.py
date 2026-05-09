"""Resampling equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["wilcox_chapter_11_equation_9"]


def wilcox_chapter_11_equation_9(x, cdf=None):
    """
    Resampling equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.

    Formula: (k = 1,...,K ), and the goal is to test the hypothesis that all C =K(J 2 − J)/2 differences are

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
        See ``moirais.fn.describe('wilcox11e9')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.11 eq.11.9
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return hypothesis_test_result(
            test_name="Resampling equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<2: insufficient data."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Resampling equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.", "p_value": float("nan")},
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
        test_name="Resampling equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.",
        statistic=float(statistic),
        pvalue=float(p_value),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Resampling equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences.", "p_value": float(p_value)},
    )


def cheatsheet():
    return "wilcox11e9: Resampling equation extracted from Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences."
