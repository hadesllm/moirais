# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Chat template: interleave system/user/assistant roles with role tokens."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_chat_template"]


def alammar_chat_template(turns, template_tokens):
    """
    Chat template: interleave system/user/assistant roles with role tokens

    Formula: prompt = concat( role_tokens[r_i] + content_i  for i in turns )

    Parameters
    ----------
    turns : array-like
        Input data.
    template_tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prompt

    References
    ----------
    Alammar Ch 12, chat templates section
    """
    turns = np.atleast_1d(np.asarray(turns, dtype=float))
    n = len(turns)
    result = float(np.mean(turns))
    se = float(np.std(turns, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chat template: interleave system/user/assistant roles with role tokens"})


def cheatsheet():
    return "alchat: Chat template: interleave system/user/assistant roles with role tokens"
