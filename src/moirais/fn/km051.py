"""Qa trigger template.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_qa_trigger_template"]


def kamath_ch3_qa_trigger_template(x, y, T, z_adv):
    """
    Qa trigger template.

    Formula: \text{Question: }[x]\text{ Context: }[y]\text{ Answer: }[T][T][T][z_{adv}]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    T : array-like
        Input data.
    z_adv : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.10, p. 105
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Qa trigger template."})


def cheatsheet():
    return "km051: Qa trigger template."
