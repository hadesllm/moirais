"""GeneralStochastic expression (auto-extracted; see reference).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["stochastic_physics_section_2_unnumbered_2"]


def stochastic_physics_section_2_unnumbered_2(x):
    """
    GeneralStochastic expression (auto-extracted; see reference).

    Formula: sites with probability 1−ps(t) = exp[−As(t)]. The movement is modeled as a biased random walk so that site s′ is

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
        See ``moirais.fn.describe('sp_s2u2')`` for the full guide.

    References
    ----------
    D'Orsogna & Perc (2015) Statistical physics of crime: A review, §2 (CRIME HOTSPOTS, unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="GeneralStochastic expression (auto-extracted; see reference).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "GeneralStochastic expression (auto-extracted; see reference)."},
    )


def cheatsheet():
    return "sp_s2u2: GeneralStochastic expression (auto-extracted; see reference)."
