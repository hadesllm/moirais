r"""Faithfulness metric.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch7_faithfulness_metric"]


def kamath_ch7_faithfulness_metric(facts):
    r"""
    Faithfulness metric.

    Formula: \mathrm{Faithfulness} = \frac{|\#\text{ facts in answer inferable from context}|}{|\#\text{ total facts in answer}|}

    Parameters
    ----------
    facts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 7, Eq 7.2, p. 300
    r"""
    facts = np.atleast_1d(np.asarray(facts, dtype=float))
    n = len(facts)
    result = float(np.mean(facts))
    se = float(np.std(facts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Faithfulness metric."})


def cheatsheet():
    return "km111: Faithfulness metric."
