# morie.fn — function file (hadesllm/morie)
"""GRU cell forward pass (Cho et al. 2014)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["gru_cell"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gru_cell(x, h_prev=None, W=None, U=None, b=None,
             hidden_size: "int | None" = None, seed: int = 0,
             deterministic_seed: "int | None" = None):
    """Single-step GRU cell.

    Gates::

        z = sigmoid(W_z x + U_z h + b_z)        update
        r = sigmoid(W_r x + U_r h + b_r)        reset
        n = tanh   (W_n x + U_n (r * h) + b_n)  candidate
        h = (1 - z) * n + z * h_prev

    ``W`` should stack ``[W_z; W_r; W_n]`` shape ``(3H, input_size)``,
    ``U`` shape ``(3H, H)``, ``b`` length ``3H``.

    Parameters
    ----------
    deterministic_seed : int or None, optional
        If given, the SHA-keyed RNG from
        :func:`morie._det_rng.from_seed` is used so Py<->R streams agree
        for the same ``(name, seed)`` pair. Overrides ``seed`` when set.

    Returns
    -------
    result : RichResult
        Keys: ``h`` / ``estimate``, ``z``, ``r``, ``n``.

    References
    ----------
    Cho, K. et al. (2014). Learning phrase representations using RNN
    encoder-decoder for statistical machine translation. *EMNLP*.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_in = x.size
    if hidden_size is None:
        if h_prev is not None:
            hidden_size = int(np.asarray(h_prev).size)
        elif W is not None:
            hidden_size = int(np.asarray(W).shape[0] // 3)
        else:
            hidden_size = n_in
    H = int(hidden_size)

    if h_prev is None:
        h_prev = np.zeros(H)
    h_prev = np.asarray(h_prev, dtype=float).ravel()

    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("grucl", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    if W is None:
        W = rng.normal(0, 0.1, size=(3 * H, n_in))
    if U is None:
        U = rng.normal(0, 0.1, size=(3 * H, H))
    if b is None:
        b = np.zeros(3 * H)
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float)

    pre = W @ x + b
    Wz, Wr, Wn = pre[0:H], pre[H:2 * H], pre[2 * H:3 * H]
    Uh = U @ h_prev
    Uz, Ur, Un = Uh[0:H], Uh[H:2 * H], Uh[2 * H:3 * H]

    z = _sigmoid(Wz + Uz)
    r = _sigmoid(Wr + Ur)
    n = np.tanh(Wn + r * Un)
    h = (1.0 - z) * n + z * h_prev

    return RichResult(
        title="GRU cell (forward)",
        summary_lines=[("hidden size", H), ("input size", n_in)],
        payload={
            "h": h,
            "estimate": h,
            "z": z, "r": r, "n": n,
            "method": "GRU cell forward",
        },
    )


# CANONICAL TEST
# With h_prev=0 and zero biases, output is deterministic given seed=0.


def cheatsheet():
    return "grucl: GRU h = (1-z)*n + z*h_prev, n=tanh(Wn+r*Un)"
