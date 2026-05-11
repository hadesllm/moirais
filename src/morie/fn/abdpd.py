# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Three-step counterfactual inference: abduction, modification, prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["abduction_modification_prediction"]


def abduction_modification_prediction(evidence, do_X, Y, scm):
    """
    Three-step counterfactual inference: abduction, modification, prediction

    Formula: Step 1: P(U|evidence); Step 2: modify SCM (do X=evidence); Step 3: predict Y in modified model

    Parameters
    ----------
    evidence : array-like
        Input data.
    do_X : array-like
        Input data.
    Y : array-like
        Input data.
    scm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'Y_counterfactual': 'float'}

    References
    ----------
    Molak Ch 2
    """
    evidence = np.asarray(evidence, dtype=float)
    n = int(evidence) if evidence.ndim == 0 else len(evidence)
    result = float(np.mean(evidence))
    se = float(np.std(evidence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Three-step counterfactual inference: abduction, modification, prediction"})


def cheatsheet():
    return "abdpd: Three-step counterfactual inference: abduction, modification, prediction"
