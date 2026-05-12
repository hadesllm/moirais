# morie.fn — function file (hadesllm/morie)
"""Perplexity (Jelinek et al. 1977)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["perplexity_metric"]


def perplexity_metric(x, base: str = "e"):
    """Perplexity of a sequence of token log-probabilities.

    Formula:  PPL = exp(-1/N * sum_i log p(x_i | x_{<i})).

    Parameters
    ----------
    x : array-like of float
        Per-token log-probabilities (natural log by default).
    base : "e" or "2"
        Log base of ``x``.  If "2", input is treated as log_2.

    Returns
    -------
    RichResult with keys: value (perplexity), nll, n.
    """
    logp = np.asarray(x, dtype=float).ravel()
    n = logp.size
    if n == 0:
        raise ValueError("Need at least one token log-prob")
    if base == "2":
        logp = logp * np.log(2.0)
    elif base != "e":
        raise ValueError("base must be 'e' or '2'")
    nll = -float(np.mean(logp))
    ppl = float(np.exp(nll))
    return RichResult(
        title="Perplexity (Jelinek 1977)",
        summary_lines=[("PPL", ppl), ("NLL/tok", nll), ("n", n)],
        payload={"value": ppl, "nll": nll, "n": n,
                 "method": "perplexity"},
        interpretation=(f"Per-token branching factor = {ppl:.4f}; "
                        f"lower is better."),
    )


def cheatsheet():
    return "pplxm(log_probs): perplexity = exp(-mean log p)"


# CANONICAL TEST
# >>> r = perplexity_metric([np.log(0.5), np.log(0.5)])
# >>> round(float(r["value"]), 4)
# 2.0
