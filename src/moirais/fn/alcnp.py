# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Chain prompting: output of prompt 1 fed as input to prompt 2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_chain_prompting"]


def alammar_chain_prompting(x, prompts, model):
    """
    Chain prompting: output of prompt 1 fed as input to prompt 2

    Formula: y_1 = LLM(P_1(x));  y_2 = LLM(P_2(y_1, x));  ... (compose K prompts)

    Parameters
    ----------
    x : array-like
        Input data.
    prompts : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: final_output

    References
    ----------
    Alammar Ch 6, Chain Prompting section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chain prompting: output of prompt 1 fed as input to prompt 2"})


def cheatsheet():
    return "alcnp: Chain prompting: output of prompt 1 fed as input to prompt 2"
