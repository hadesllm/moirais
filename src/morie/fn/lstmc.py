# morie.fn -- function file (hadesllm/morie)
"""LSTM cell forward pass."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["lstm_cell"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm_cell(x, h_prev=None, c_prev=None, W=None, U=None, b=None,
              hidden_size: "int | None" = None, seed: int = 0,
              deterministic_seed: "int | None" = None):
    """Single-step LSTM cell (Hochreiter & Schmidhuber 1997).

    Gates::

        f = sigmoid(W_f x + U_f h + b_f)        forget
        i = sigmoid(W_i x + U_i h + b_i)        input
        g = tanh   (W_g x + U_g h + b_g)        candidate
        o = sigmoid(W_o x + U_o h + b_o)        output

        c = f * c_prev + i * g
        h = o * tanh(c)

    ``W`` should be shape ``(4H, input_size)`` stacking
    ``[W_i; W_f; W_g; W_o]``; ``U`` shape ``(4H, H)``; ``b`` length
    ``4H``.  If any of them are ``None``, small deterministic random
    parameters are generated for the smoke test.

    Parameters
    ----------
    deterministic_seed : int or None, optional
        If given, the SHA-keyed RNG from
        :func:`morie._det_rng.from_seed` is used so Py<->R streams agree
        for the same ``(name, seed)`` pair. Overrides ``seed`` when set.

    Returns
    -------
    result : RichResult
        Keys: ``h``, ``c``, ``estimate`` (= ``h``), gate activations.

    References
    ----------
    Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
    *Neural Computation*, 9(8), 1735-1780.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_in = x.size
    if hidden_size is None:
        if h_prev is not None:
            hidden_size = int(np.asarray(h_prev).size)
        elif W is not None:
            hidden_size = int(np.asarray(W).shape[0] // 4)
        else:
            hidden_size = n_in
    H = int(hidden_size)

    if h_prev is None:
        h_prev = np.zeros(H)
    if c_prev is None:
        c_prev = np.zeros(H)
    h_prev = np.asarray(h_prev, dtype=float).ravel()
    c_prev = np.asarray(c_prev, dtype=float).ravel()

    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("lstmc", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    if W is None:
        W = rng.normal(0, 0.1, size=(4 * H, n_in))
    if U is None:
        U = rng.normal(0, 0.1, size=(4 * H, H))
    if b is None:
        b = np.zeros(4 * H)
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float)

    gates = W @ x + U @ h_prev + b  # (4H,)
    i = _sigmoid(gates[0:H])
    f = _sigmoid(gates[H:2 * H])
    g = np.tanh(gates[2 * H:3 * H])
    o = _sigmoid(gates[3 * H:4 * H])

    c = f * c_prev + i * g
    h = o * np.tanh(c)

    return RichResult(
        title="LSTM cell (forward)",
        summary_lines=[("hidden size", H), ("input size", n_in)],
        payload={
            "h": h,
            "c": c,
            "estimate": h,
            "i": i, "f": f, "g": g, "o": o,
            "method": "LSTM cell forward",
        },
    )


# CANONICAL TEST
# With h_prev=c_prev=0 and zero biases, well-defined deterministic output
# given seed=0.


def cheatsheet():
    return "lstmc: LSTM cell f,i,o=sigmoid(...), c=f*c+i*g, h=o*tanh(c)"
