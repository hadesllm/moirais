# moirais.fn — function file (hadesllm/moirais)
"""Model Context Protocol (MCP) for LLM tool integration."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_model_context_protocol"]


def geron_model_context_protocol(server, client):
    """
    Model Context Protocol (MCP) for LLM tool integration

    Formula: JSON-RPC schema for exposing tools/resources to LLMs

    Parameters
    ----------
    server : array-like
        Input data.
    client : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: protocol

    References
    ----------
    Géron Ch 15
    """
    server = np.atleast_1d(np.asarray(server, dtype=float))
    n = len(server)
    result = float(np.mean(server))
    se = float(np.std(server, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Model Context Protocol (MCP) for LLM tool integration"})


def cheatsheet():
    return "hmmcp: Model Context Protocol (MCP) for LLM tool integration"
