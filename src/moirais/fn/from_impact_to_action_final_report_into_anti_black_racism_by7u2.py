"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["from_impact_to_action_final_report_into_anti_black_racism_by_chapter_7_unnumbered_2"]


def from_impact_to_action_final_report_into_anti_black_racism_by_chapter_7_unnumbered_2(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: [77] TPSB: June 22, 2022 (22 June 2022) at 9, online (pdf): https://tpsb.ca/jdownloads-categories?task=download.send&id=747&catid=62&m=0.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``moirais.fn.describe('from_impact_to_action_final_report_into_anti_black_racism_by7u2')`` for the full guide.

    References
    ----------
    From Impact to Action- Final report into anti-Black racism by the Toronto Police Service, ch.7 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "from_impact_to_action_final_report_into_anti_black_racism_by7u2: CentralTendency expression (auto-extracted; see ref)."
