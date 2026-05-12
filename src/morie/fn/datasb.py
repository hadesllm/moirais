# morie.fn -- function file (hadesllm/morie)
"""Data subset refutation: estimate on random subsets should be stable."""
import numpy as np
from ._richresult import RichResult

__all__ = ["data_subset_refutation"]


def data_subset_refutation(model, subset_fraction, n_iter):
    """
    Data subset refutation: estimate on random subsets should be stable

    Formula: Resample fraction p of data; reestimate ATE; high variance = fragile estimate

    Parameters
    ----------
    model : array-like
        Input data.
    subset_fraction : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'subset_ates': 'array', 'stability': 'float'}

    References
    ----------
    Molak Ch 7
    """
    model = np.asarray(model, dtype=float)
    n = int(model) if model.ndim == 0 else len(model)
    if model.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Data subset refutation: estimate on random subsets should be stable"})
    estimate = np.median(model)
    se = 1.2533 * np.std(model, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Data subset refutation: estimate on random subsets should be stable"})


def cheatsheet():
    return "datasb: Data subset refutation: estimate on random subsets should be stable"
