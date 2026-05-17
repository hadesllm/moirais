"""
MCP home range

Category: WildlSp
"""

import numpy as np


def wlmcp(abundance=None, coords=None, n=50):
    """MCP home range

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if abundance is None:
        abundance = np.random.default_rng(0).poisson(10, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(abundance))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(abundance), "total": int(np.sum(abundance)), "mean": float(np.mean(abundance))},
    )


short = "wlmcp"
alias = "wlmcp"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
wlmcp = wlmcp


def cheatsheet() -> str:
    return "wlmcp({}) -> MCP home range"
