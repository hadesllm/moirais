"""GeneralStochastic equation from D'Orsogna & Perc review.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["stochastic_physics_equation_6"]


def stochastic_physics_equation_6(x):
    """
    GeneralStochastic equation from D'Orsogna & Perc review.

    Formula: [EQ] 1 + exp(Ps′−Ps)/K], (6)

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
        See ``morie.fn.describe('sp_eq6')`` for the full guide.

    References
    ----------
    D'Orsogna & Perc (2015) Statistical physics of crime: A review, §3 (SELF-EXCITING POINT PROCESS MODELING), eq.(6)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="GeneralStochastic equation from D'Orsogna & Perc review.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "GeneralStochastic equation from D'Orsogna & Perc review."},
    )


def cheatsheet():
    return "sp_eq6: GeneralStochastic equation from D'Orsogna & Perc review."
