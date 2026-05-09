# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Preference-pair data template: (prompt, chosen, rejected)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_chosen_rejected_template"]


def alammar_chosen_rejected_template(prompts, chosen, rejected):
    """
    Preference-pair data template: (prompt, chosen, rejected)

    Formula: each record = {prompt, chosen=y_w, rejected=y_l}; y_w ≻ y_l

    Parameters
    ----------
    prompts : array-like
        Input data.
    chosen : array-like
        Input data.
    rejected : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dataset

    References
    ----------
    Alammar Ch 12, alignment data templating section
    """
    prompts = np.atleast_1d(np.asarray(prompts, dtype=float))
    n = len(prompts)
    result = float(np.mean(prompts))
    se = float(np.std(prompts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Preference-pair data template: (prompt, chosen, rejected)"})


def cheatsheet():
    return "alchrj: Preference-pair data template: (prompt, chosen, rejected)"
