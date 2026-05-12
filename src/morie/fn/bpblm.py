# morie.fn -- function file (hadesllm/morie)
"""Bits-per-byte evaluation metric (Gao et al. 2020, The Pile)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["bits_per_byte"]


def bits_per_byte(x, n_bytes: int | None = None):
    """Bits-per-byte (BPB) -- model-size-independent LM metric.

    Formula:  BPB = NLL_nats / (N_tokens * ln(2) * bytes_per_token)
            = total_NLL_nats / (n_bytes * ln(2))

    Parameters
    ----------
    x : array-like of float
        Per-token negative log-likelihoods (in nats).
    n_bytes : int, optional
        Total number of bytes the tokens correspond to.  If None,
        assumes 1 byte per token (BPB == bits-per-token).

    Returns
    -------
    RichResult with keys: value (BPB), nll_nats, n_tokens, n_bytes.
    """
    nll = np.asarray(x, dtype=float).ravel()
    if nll.size == 0:
        raise ValueError("Need at least one token NLL")
    total_nats = float(np.sum(nll))
    n_tok = int(nll.size)
    nb = int(n_bytes) if n_bytes is not None else n_tok
    if nb <= 0:
        raise ValueError("n_bytes must be > 0")
    bpb = total_nats / (nb * np.log(2.0))
    return RichResult(
        title="Bits per Byte (Gao 2020)",
        summary_lines=[("BPB", bpb), ("n_tokens", n_tok), ("n_bytes", nb)],
        payload={"value": float(bpb), "nll_nats": total_nats,
                 "n_tokens": n_tok, "n_bytes": nb,
                 "method": "BPB"},
    )


def cheatsheet():
    return "bpblm(token_nll_nats, n_bytes): bits per byte"


# CANONICAL TEST
# >>> r = bits_per_byte([np.log(2.0)] * 4, n_bytes=4)
# >>> bool(np.isclose(float(r["value"]), 1.0))
# True
