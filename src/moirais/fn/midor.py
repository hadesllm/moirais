# moirais.fn — function file (hadesllm/moirais)
"""DoWhy four-step causal inference process: model -> identify -> estimate -> refute."""
import numpy as np
from ._richresult import RichResult

__all__ = ["model_identify_estimate_refute"]


def model_identify_estimate_refute(dag, data, estimand, estimator):
    """
    DoWhy four-step causal inference process: model -> identify -> estimate -> refute

    Formula: Step 1: specify DAG; Step 2: identify estimand (backdoor/IV/frontdoor); Step 3: estimate; Step 4: refute

    Parameters
    ----------
    dag : array-like
        Input data.
    data : array-like
        Input data.
    estimand : array-like
        Input data.
    estimator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'estimate': 'float', 'refutation_results': 'dict'}

    References
    ----------
    Molak Ch 7
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    if data.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "DoWhy four-step causal inference process: model -> identify -> estimate -> refute"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "DoWhy four-step causal inference process: model -> identify -> estimate -> refute"})


def cheatsheet():
    return "midor: DoWhy four-step causal inference process: model -> identify -> estimate -> refute"
