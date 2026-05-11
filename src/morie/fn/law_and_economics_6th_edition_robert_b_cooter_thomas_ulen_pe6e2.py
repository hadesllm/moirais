"""GeneralStatistics equation extracted from Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_6_equation_2"]


def law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe_chapter_6_equation_2(x, cdf=None):
    """
    GeneralStatistics equation extracted from Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive.

    Formula: first derivative of Equation 6.1 with respect to pre

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
        See ``morie.fn.describe('law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe6e2')`` for the full guide.

    References
    ----------
    Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive, ch.6 eq.6.2
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return hypothesis_test_result(
            test_name="GeneralStatistics equation extracted from Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive.",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<2: insufficient data."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "GeneralStatistics equation extracted from Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive.", "p_value": float("nan")},
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
        test_name="GeneralStatistics equation extracted from Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive.",
        statistic=float(statistic),
        pvalue=float(p_value),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "GeneralStatistics equation extracted from Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive.", "p_value": float(p_value)},
    )


def cheatsheet():
    return "law_and_economics_6th_edition_robert_b_cooter_thomas_ulen_pe6e2: GeneralStatistics equation extracted from Law and Economics, 6th Edition -- Robert B. Cooter, Thomas Ulen -- Pearson Series in Economics, 6th Edition, 2011 -- Prentice Hall -- 9780132540650 -- 4bc55da23884b7603280fecc527f4743 -- Anna’s Archive."
