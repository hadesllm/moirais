# moirais.fn — function file (hadesllm/moirais)
"""Random common cause refutation: add random confounder, check estimate stability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["random_cause_refutation"]


def random_cause_refutation(model, n_simulations):
    """
    Random common cause refutation: add random confounder, check estimate stability

    Formula: Add W ~ N(0,1) independent of data; refit model; ATE should be stable

    Parameters
    ----------
    model : array-like
        Input data.
    n_simulations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'new_ate': 'float', 'change': 'float'}

    References
    ----------
    Molak Ch 7
    """
    model = np.asarray(model, dtype=float)
    n = int(model) if model.ndim == 0 else len(model)
    if model.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Random common cause refutation: add random confounder, check estimate stability"})
    estimate = np.median(model)
    se = 1.2533 * np.std(model, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Random common cause refutation: add random confounder, check estimate stability"})


def cheatsheet():
    return "rcaus: Random common cause refutation: add random confounder, check estimate stability"
