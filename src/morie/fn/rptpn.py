# morie.fn -- function file (hadesllm/morie)
"""Repetition penalty for generation (Keskar et al. 2019, CTRL)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["repetition_penalty"]


def repetition_penalty(x, generated, alpha: float = 1.2):
    """CTRL-style repetition penalty.

    Formula (Keskar 2019):
        logit[t] /= alpha    if t in generated and logit[t] > 0
        logit[t] *= alpha    if t in generated and logit[t] <= 0

    The asymmetric form ensures the penalty always reduces the
    likelihood of repeating ``t`` regardless of the logit sign.

    Parameters
    ----------
    x : array-like of logits, shape (V,).
    generated : iterable of int
        Token ids already in the generated prefix.
    alpha : float
        Repetition penalty strength.  alpha > 1 discourages repeats;
        alpha == 1 is a no-op.

    Returns
    -------
    RichResult with keys: tensor (penalised logits), penalised_idx.
    """
    z = np.asarray(x, dtype=float).copy()
    if alpha == 1.0:
        return RichResult(
            title="Repetition Penalty (Keskar 2019)",
            payload={"tensor": z, "penalised_idx": np.array([], int),
                     "alpha": alpha, "method": "rep-penalty"},
        )
    idx = np.unique(np.asarray(list(generated), dtype=int))
    idx = idx[(idx >= 0) & (idx < z.size)]
    sel = z[idx]
    z[idx] = np.where(sel > 0, sel / alpha, sel * alpha)
    return RichResult(
        title="Repetition Penalty (Keskar 2019)",
        summary_lines=[("alpha", alpha), ("n_penalised", idx.size)],
        payload={"tensor": z, "penalised_idx": idx, "alpha": alpha,
                 "method": "rep-penalty"},
    )


def cheatsheet():
    return "rptpn(logits, generated, alpha): CTRL-style repetition penalty"


# CANONICAL TEST
# >>> r = repetition_penalty([2.0, -2.0, 1.0], generated=[0, 1], alpha=2.0)
# >>> bool(np.allclose(r["tensor"], [1.0, -4.0, 1.0]))
# True
