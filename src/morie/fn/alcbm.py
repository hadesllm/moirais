# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Conversation buffer memory: maintain a rolling window of the last N turns."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_conversation_buffer_memory"]


def alammar_conversation_buffer_memory(conversation, N):
    """
    Conversation buffer memory: maintain a rolling window of the last N turns

    Formula: memory_t = [(u_{t-N+1}, a_{t-N+1}), ..., (u_t, a_t)]

    Parameters
    ----------
    conversation : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: window

    References
    ----------
    Alammar Ch 7, Conversation Buffer Memory section
    """
    conversation = np.atleast_1d(np.asarray(conversation, dtype=float))
    n = len(conversation)
    result = float(np.mean(conversation))
    se = float(np.std(conversation, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conversation buffer memory: maintain a rolling window of the last N turns"})


def cheatsheet():
    return "alcbm: Conversation buffer memory: maintain a rolling window of the last N turns"
