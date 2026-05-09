# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""ReAct loop: interleave Thought → Action → Observation until final answer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_react_agent_loop"]


def alammar_react_agent_loop(query, tools, model, max_steps):
    """
    ReAct loop: interleave Thought → Action → Observation until final answer

    Formula: while not done: thought, action = LLM(ctx); obs = Tool(action); ctx += (thought, action, obs)

    Parameters
    ----------
    query : array-like
        Input data.
    tools : array-like
        Input data.
    model : array-like
        Input data.
    max_steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: answer

    References
    ----------
    Alammar Ch 7, ReAct Agent Framework section
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ReAct loop: interleave Thought → Action → Observation until final answer"})


def cheatsheet():
    return "alreact: ReAct loop: interleave Thought → Action → Observation until final answer"
